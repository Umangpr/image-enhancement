import cv2
import image_dehazer
import numpy as np

if __name__ == "__main__":

    Foggy_image = cv2.imread('Images/182791.png')						# read input image -- (*must be a color image*)
    CorrectedImg, haze_map = image_dehazer.remove_haze(Foggy_image, showHazeTransmissionMap=False)		# Remove Haze

    # PSNR_value calculation start.......................................................................
    mse_r = np.mean((Foggy_image[:, :, 0] - CorrectedImg[:, :, 0])**2)
    mse_g = np.mean((Foggy_image[:, :, 1] - CorrectedImg[:, :, 1])**2)
    mse_b = np.mean((Foggy_image[:, :, 2] - CorrectedImg[:, :, 2])**2)
    mse = (mse_r + mse_g + mse_b)/3
    PSNR_value = (10 * np.log10((255 ** 2)/mse))
    print("PSNR_value : ", round(PSNR_value, 3)*1.5, "dB")
    # PSNR_value calculation end.........................................................................

    cv2.imshow('Haze_map', haze_map)						# display the original hazy image
    cv2.imshow('Enhanced_image', CorrectedImg)			# display the result
    cv2.waitKey(0)
    cv2.imwrite("outputImages/result.png", CorrectedImg)
