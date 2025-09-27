from preprocessing import crop_pcba_to_jpg

file_output_name = "test-image-processed-output.jpg"
cropped_image = crop_pcba_to_jpg("test-image-unprocessed.jpg",file_output_name, prefer="auto" )

print("Saved:", cropped_image)