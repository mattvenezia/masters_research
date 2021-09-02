(minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(blur)
# Draw circle around location of the maxVal pixel
cv2.circle(crop_img, maxLoc, 5, (255, 0, 0), 2)
# display the results of the naive attempt
cv2.imshow("Naive", crop_img)
cv2.waitKey(0)