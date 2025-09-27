from preprocessing import crop_pcba_to_jpg

file_output_name = "test-processed-output.jpg"
cropped_image = crop_pcba_to_jpg("test.jpg",file_output_name, prefer="auto" )

print("Saved:", cropped_image)
