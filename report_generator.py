"""
report_generator.py - PCBA Defect Report Generator

PURPOSE:
Generates comprehensive defect reports from inference results.
Creates both JSON and HTML reports with defect locations, severity, and recommendations.

INPUTS:
- Defect inference results JSON
- Pipeline run directory (for additional context)

OUTPUTS:
- Structured JSON defect report
- HTML report with visualizations
- Summary statistics

DEPENDENCIES:
- json
- datetime

USAGE:
python3 report_generator.py --results predictions.json --run-dir results/ --output report.json --html report.html
"""

import json
import argparse
import sys
from pathlib import Path
from datetime import datetime
import math

def load_inference_results(results_file):
    """
    Load defect inference results from JSON file
    
    Args:
        results_file: Path to inference results JSON
    
    Returns:
        dict: Inference results data
    """
    if not Path(results_file).exists():
        raise FileNotFoundError(f"Results file not found: {results_file}")
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    print(f"Loaded results for {results['metadata']['total_patches_processed']} patches")
    return results

def classify_defect_severity(defect_type, confidence):
    """
    Classify defect severity based on type and confidence
    
    Args:
        defect_type: Type of defect detected
        confidence: Prediction confidence
    
    Returns:
        str: Severity level ('CRITICAL', 'HIGH', 'MEDIUM', 'LOW')
    """
    # Define severity rules
    critical_defects = ['has_missing']  # Missing components are critical
    high_defects = ['has_solder', 'has_rotate']  # Solder issues and misalignment
    medium_defects = ['has_dirt']  # Contamination
    
    # Base severity on defect type
    if defect_type in critical_defects:
        base_severity = 'CRITICAL'
    elif defect_type in high_defects:
        base_severity = 'HIGH'
    elif defect_type in medium_defects:
        base_severity = 'MEDIUM'
    else:
        base_severity = 'LOW'
    
    # Adjust based on confidence
    if confidence < 0.6:
        # Lower confidence reduces severity
        severity_map = {'CRITICAL': 'HIGH', 'HIGH': 'MEDIUM', 'MEDIUM': 'LOW', 'LOW': 'LOW'}
        return severity_map[base_severity]
    elif confidence > 0.9:
        # Very high confidence maintains or increases severity
        return base_severity
    else:
        return base_severity

def extract_defect_locations(patch_results):
    """
    Extract defect locations with detailed information
    
    Args:
        patch_results: List of patch inference results
    
    Returns:
        list: Defect location data
    """
    defect_locations = []
    
    for result in patch_results:
        predictions = result['predictions']
        active_labels = predictions['active_labels']
        
        # Find defect labels
        defect_labels = [label for label in active_labels 
                        if label in ['has_dirt', 'has_missing', 'has_rotate', 'has_solder']]
        
        if defect_labels:
            # Get patch coordinates
            coords = result['patch_coordinates']
            center_x = (coords[0] + coords[2]) // 2
            center_y = (coords[1] + coords[3]) // 2
            
            # Find associated components
            associated_components = []
            for comp in result['overlapping_components']:
                associated_components.append({
                    'class': comp['class'],
                    'confidence': comp['confidence'],
                    'detection_id': comp.get('detection_id', 'unknown')
                })
            
            # Create defect entry for each defect type found
            for defect_type in defect_labels:
                defect_confidence = predictions['all_predictions'].get(defect_type, 0.0)
                severity = classify_defect_severity(defect_type, defect_confidence)
                
                defect_location = {
                    'defect_id': f"D{len(defect_locations) + 1:03d}",
                    'defect_type': defect_type.replace('has_', ''),
                    'severity': severity,
                    'confidence': round(defect_confidence, 3),
                    'location': {
                        'patch_index': result['patch_index'],
                        'center_coordinates': [center_x, center_y],
                        'bounding_box': coords,
                        'quadrant': get_board_quadrant(center_x, center_y, coords)
                    },
                    'associated_components': associated_components,
                    'description': get_defect_description(defect_type, associated_components)
                }
                
                defect_locations.append(defect_location)
    
    # Sort by severity and confidence
    severity_order = {'CRITICAL': 0, 'HIGH': 1, 'MEDIUM': 2, 'LOW': 3}
    defect_locations.sort(key=lambda x: (severity_order[x['severity']], -x['confidence']))
    
    return defect_locations

def get_board_quadrant(x, y, image_bounds):
    """
    Determine which quadrant of the board the defect is in
    
    Args:
        x, y: Coordinates
        image_bounds: [x1, y1, x2, y2] bounds
    
    Returns:
        str: Quadrant description
    """
    center_x = (image_bounds[0] + image_bounds[2]) // 2
    center_y = (image_bounds[1] + image_bounds[3]) // 2
    
    if x < center_x and y < center_y:
        return "Top-Left"
    elif x >= center_x and y < center_y:
        return "Top-Right"
    elif x < center_x and y >= center_y:
        return "Bottom-Left"
    else:
        return "Bottom-Right"

def get_defect_description(defect_type, associated_components):
    """
    Generate human-readable defect description
    
    Args:
        defect_type: Type of defect
        associated_components: List of associated components
    
    Returns:
        str: Description of the defect
    """
    defect_descriptions = {
        'has_missing': 'Missing component',
        'has_solder': 'Solder defect (bridge, insufficient, or excess)',
        'has_rotate': 'Component rotation or alignment issue',
        'has_dirt': 'Contamination or foreign material'
    }
    
    base_description = defect_descriptions.get(defect_type, 'Unknown defect type')
    
    if associated_components:
        component_names = [comp['class'] for comp in associated_components]
        unique_components = list(set(component_names))
        
        if len(unique_components) == 1:
            component_str = unique_components[0]
        elif len(unique_components) == 2:
            component_str = f"{unique_components[0]} and {unique_components[1]}"
        else:
            component_str = f"{', '.join(unique_components[:-1])}, and {unique_components[-1]}"
        
        return f"{base_description} affecting {component_str}"
    else:
        return f"{base_description} in background area"

def generate_summary_statistics(inference_results, defect_locations):
    """
    Generate summary statistics for the report
    
    Args:
        inference_results: Full inference results
        defect_locations: Extracted defect locations
    
    Returns:
        dict: Summary statistics
    """
    analysis = inference_results['analysis']
    
    # Defect severity breakdown
    severity_counts = {'CRITICAL': 0, 'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
    defect_type_counts = {}
    
    for defect in defect_locations:
        severity_counts[defect['severity']] += 1
        defect_type = defect['defect_type']
        defect_type_counts[defect_type] = defect_type_counts.get(defect_type, 0) + 1
    
    # Calculate board status
    total_defects = len(defect_locations)
    critical_defects = severity_counts['CRITICAL']
    high_defects = severity_counts['HIGH']
    
    if critical_defects > 0:
        board_status = "REJECT"
        status_reason = f"{critical_defects} critical defect(s) found"
    elif high_defects > 2:
        board_status = "REJECT"  
        status_reason = f"{high_defects} high-severity defects found"
    elif total_defects > 5:
        board_status = "REVIEW"
        status_reason = f"{total_defects} total defects require review"
    elif total_defects > 0:
        board_status = "REVIEW"
        status_reason = f"{total_defects} defect(s) detected"
    else:
        board_status = "PASS"
        status_reason = "No defects detected"
    
    summary = {
        'board_status': board_status,
        'status_reason': status_reason,
        'total_defects': total_defects,
        'severity_breakdown': severity_counts,
        'defect_type_breakdown': defect_type_counts,
        'patches_analyzed': analysis['total_patches'],
        'defect_density': round(total_defects / analysis['total_patches'] * 100, 2),
        'components_detected': len(analysis.get('component_types_detected', {})),
        'high_confidence_defects': analysis.get('high_confidence_defects', 0)
    }
    
    return summary

def generate_recommendations(defect_locations, summary_stats):
    """
    Generate actionable recommendations based on defects found
    
    Args:
        defect_locations: List of defect locations
        summary_stats: Summary statistics
    
    Returns:
        list: List of recommendations
    """
    recommendations = []
    
    # Critical defect recommendations
    critical_defects = [d for d in defect_locations if d['severity'] == 'CRITICAL']
    if critical_defects:
        missing_components = [d for d in critical_defects if d['defect_type'] == 'missing']
        if missing_components:
            recommendations.append({
                'priority': 'CRITICAL',
                'action': 'Component Placement',
                'description': f"Install {len(missing_components)} missing component(s)",
                'locations': [d['location']['center_coordinates'] for d in missing_components]
            })
    
    # Solder defect recommendations
    solder_defects = [d for d in defect_locations if d['defect_type'] == 'solder']
    if solder_defects:
        recommendations.append({
            'priority': 'HIGH',
            'action': 'Solder Rework',
            'description': f"Rework {len(solder_defects)} solder joint(s)",
            'locations': [d['location']['center_coordinates'] for d in solder_defects]
        })
    
    # Rotation/alignment recommendations
    rotation_defects = [d for d in defect_locations if d['defect_type'] == 'rotate']
    if rotation_defects:
        recommendations.append({
            'priority': 'HIGH',
            'action': 'Component Realignment',
            'description': f"Realign {len(rotation_defects)} component(s)",
            'locations': [d['location']['center_coordinates'] for d in rotation_defects]
        })
    
    # Contamination recommendations
    dirt_defects = [d for d in defect_locations if d['defect_type'] == 'dirt']
    if dirt_defects:
        recommendations.append({
            'priority': 'MEDIUM',
            'action': 'Cleaning',
            'description': f"Clean {len(dirt_defects)} contaminated area(s)",
            'locations': [d['location']['center_coordinates'] for d in dirt_defects]
        })
    
    # Overall recommendations
    if summary_stats['board_status'] == 'REJECT':
        recommendations.insert(0, {
            'priority': 'CRITICAL',
            'action': 'Board Rejection',
            'description': 'Board fails quality standards and requires rework',
            'locations': []
        })
    elif summary_stats['defect_density'] > 5:
        recommendations.append({
            'priority': 'MEDIUM',
            'action': 'Process Review',
            'description': 'High defect density suggests process optimization needed',
            'locations': []
        })
    
    return recommendations

def create_json_report(inference_results, defect_locations, summary_stats, recommendations):
    """
    Create structured JSON report
    
    Args:
        inference_results: Original inference results
        defect_locations: Extracted defect locations
        summary_stats: Summary statistics
        recommendations: Generated recommendations
    
    Returns:
        dict: Complete JSON report
    """
    report = {
        'report_metadata': {
            'report_id': f"PCBA_DEFECT_REPORT_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'generation_timestamp': datetime.now().isoformat(),
            'model_info': {
                'model_file': inference_results['metadata']['model_file'],
                'confidence_threshold': inference_results['metadata']['confidence_threshold']
            },
            'source_data': {
                'patches_directory': inference_results['metadata']['patches_directory'],
                'total_patches_analyzed': inference_results['metadata']['total_patches_processed']
            }
        },
        'summary': summary_stats,
        'defect_locations': defect_locations,
        'recommendations': recommendations,
        'raw_analysis': inference_results['analysis']
    }
    
    return report

def create_html_report(json_report):
    """
    Create HTML report for better visualization
    
    Args:
        json_report: Complete JSON report data
    
    Returns:
        str: HTML report content
    """
    summary = json_report['summary']
    defects = json_report['defect_locations']
    recommendations = json_report['recommendations']
    
    # Determine status color
    status_colors = {
        'PASS': '#28a745',
        'REVIEW': '#ffc107', 
        'REJECT': '#dc3545'
    }
    status_color = status_colors.get(summary['board_status'], '#6c757d')
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>PCBA Defect Detection Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f8f9fa; }}
            .header {{ background-color: {status_color}; color: white; padding: 20px; border-radius: 10px; text-align: center; }}
            .section {{ background-color: white; margin: 20px 0; padding: 20px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            .defect-item {{ border-left: 4px solid #dc3545; padding: 15px; margin: 10px 0; background-color: #f8f9fa; }}
            .defect-critical {{ border-left-color: #dc3545; }}
            .defect-high {{ border-left-color: #fd7e14; }}
            .defect-medium {{ border-left-color: #ffc107; }}
            .defect-low {{ border-left-color: #28a745; }}
            .recommendation {{ border-left: 4px solid #007bff; padding: 15px; margin: 10px 0; background-color: #e7f3ff; }}
            .metric {{ display: inline-block; margin: 10px 20px; text-align: center; }}
            .metric-value {{ font-size: 2em; font-weight: bold; color: {status_color}; }}
            .metric-label {{ font-size: 0.9em; color: #6c757d; }}
            table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
            th, td {{ border: 1px solid #dee2e6; padding: 12px; text-align: left; }}
            th {{ background-color: #e9ecef; }}
            .coordinates {{ font-family: monospace; background-color: #f8f9fa; padding: 2px 4px; border-radius: 3px; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>PCBA Defect Detection Report</h1>
            <h2>Status: {summary['board_status']}</h2>
            <p>{summary['status_reason']}</p>
            <p>Generated: {json_report['report_metadata']['generation_timestamp']}</p>
        </div>
        
        <div class="section">
            <h2>Summary Statistics</h2>
            <div style="text-align: center;">
                <div class="metric">
                    <div class="metric-value">{summary['total_defects']}</div>
                    <div class="metric-label">Total Defects</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{summary['defect_density']}%</div>
                    <div class="metric-label">Defect Density</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{summary['components_detected']}</div>
                    <div class="metric-label">Components Detected</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{summary['patches_analyzed']}</div>
                    <div class="metric-label">Patches Analyzed</div>
                </div>
            </div>
            
            <h3>Severity Breakdown</h3>
            <table>
                <tr>
                    <th>Severity</th>
                    <th>Count</th>
                    <th>Percentage</th>
                </tr>
    """
    
    for severity, count in summary['severity_breakdown'].items():
        if count > 0:
            percentage = (count / summary['total_defects'] * 100) if summary['total_defects'] > 0 else 0
            html_content += f"""
                <tr>
                    <td>{severity}</td>
                    <td>{count}</td>
                    <td>{percentage:.1f}%</td>
                </tr>
            """
    
    html_content += """
            </table>
        </div>
    """
    
    # Defect locations section
    if defects:
        html_content += """
        <div class="section">
            <h2>Defect Locations</h2>
        """
        
        for defect in defects:
            severity_class = f"defect-{defect['severity'].lower()}"
            coords = defect['location']['center_coordinates']
            
            html_content += f"""
            <div class="defect-item {severity_class}">
                <h4>{defect['defect_id']}: {defect['defect_type'].title()} ({defect['severity']})</h4>
                <p><strong>Description:</strong> {defect['description']}</p>
                <p><strong>Confidence:</strong> {defect['confidence']:.1%}</p>
                <p><strong>Location:</strong> <span class="coordinates">({coords[0]}, {coords[1]})</span> - {defect['location']['quadrant']}</p>
            """
            
            if defect['associated_components']:
                html_content += "<p><strong>Associated Components:</strong> "
                comp_list = [f"{comp['class']} (conf: {comp['confidence']:.2f})" 
                           for comp in defect['associated_components']]
                html_content += ", ".join(comp_list) + "</p>"
            
            html_content += "</div>"
        
        html_content += "</div>"
    
    # Recommendations section
    if recommendations:
        html_content += """
        <div class="section">
            <h2>Recommendations</h2>
        """
        
        for rec in recommendations:
            html_content += f"""
            <div class="recommendation">
                <h4>{rec['priority']} Priority: {rec['action']}</h4>
                <p>{rec['description']}</p>
            """
            
            if rec['locations']:
                html_content += "<p><strong>Locations:</strong> "
                location_strs = [f"({loc[0]}, {loc[1]})" for loc in rec['locations']]
                html_content += ", ".join(location_strs) + "</p>"
            
            html_content += "</div>"
        
        html_content += "</div>"
    
    html_content += """
    </body>
    </html>
    """
    
    return html_content

def main():
    """Main report generation function"""
    parser = argparse.ArgumentParser(description="Generate PCBA defect detection reports")
    parser.add_argument("--results", required=True, help="Defect inference results JSON file")
    parser.add_argument("--run-dir", help="Pipeline run directory (optional)")
    parser.add_argument("--output", required=True, help="Output JSON report file")
    parser.add_argument("--html", help="Optional HTML report file")
    
    args = parser.parse_args()
    
    try:
        # Load inference results
        inference_results = load_inference_results(args.results)
        
        # Extract defect locations
        defect_locations = extract_defect_locations(inference_results['patch_results'])
        
        # Generate summary statistics
        summary_stats = generate_summary_statistics(inference_results, defect_locations)
        
        # Generate recommendations
        recommendations = generate_recommendations(defect_locations, summary_stats)
        
        # Create JSON report
        json_report = create_json_report(
            inference_results, defect_locations, summary_stats, recommendations
        )
        
        # Save JSON report
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(json_report, f, indent=2)
        
        # Create HTML report if requested
        if args.html:
            html_content = create_html_report(json_report)
            
            html_path = Path(args.html)
            html_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(html_path, 'w') as f:
                f.write(html_content)
            
            print(f"HTML report saved to: {html_path}")
        
        # Print summary
        print(f"\nDefect Report Summary:")
        print(f"Board Status: {summary_stats['board_status']}")
        print(f"Total Defects: {summary_stats['total_defects']}")
        print(f"Critical: {summary_stats['severity_breakdown']['CRITICAL']}")
        print(f"High: {summary_stats['severity_breakdown']['HIGH']}")
        print(f"Medium: {summary_stats['severity_breakdown']['MEDIUM']}")
        print(f"Low: {summary_stats['severity_breakdown']['LOW']}")
        print(f"Recommendations: {len(recommendations)}")
        print(f"JSON report saved to: {output_path}")
        
        return 0
        
    except Exception as e:
        print(f"Report generation failed: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())