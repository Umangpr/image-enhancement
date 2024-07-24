# # importing libraries....
# import cv2
# import numpy as np
# import copy


# class image_dehazer():

#     def _init_(self, airlightEstimation_windowSze=15, boundaryConstraint_windowSze=3, C0=20, C1=300,
#                  regularize_lambda=0.1, sigma=0.5, delta=0.85, showHazeTransmissionMap=True):
#         self.airlightEstimation_windowSze = airlightEstimation_windowSze
#         self.boundaryConstraint_windowSze = boundaryConstraint_windowSze
#         self.C0 = C0
#         self.C1 = C1
#         self.regularize_lambda = regularize_lambda
#         self.sigma = sigma
#         self.delta = delta
#         self.showHazeTransmissionMap = showHazeTransmissionMap
#         self._A = []
#         self._transmission = []
#         self._WFun = []

#     def air_light_estimation(self, haze_img):
#         # Check if the image is a color image or gray scale...
#         if len(haze_img.shape == 3):
#             # if it is a color image , process each color channel separately.
#             for ch in range(len(haze_img.shape)):
#               kernel = np.ones((self.airlightEstimation_windowSze, self.airlightEstimation_windowSze), np.uint8)
#               min_img = cv2.erode(haze_img[:, :, ch], kernel)
#                 # Find the maximum value in image and append it to _A
#               self._A.append(int(min_img.max()))
#         else:
#             # if it is a gray scale, use the same process without iterating through channel.
#             kernel = np.ones((self.airlightEstimation_windowSze, self.airlightEstimation_windowSze), np.uint8)
#             min_img = cv2.erode(haze_img, kernel)
#             self._A.append(int(min_img.max()))

#     def boundary_constraint(self, haze_img):
#         # Check if the image is a color image or gray scale...
#         if len(haze_img.shape == 3):
#             # For color images, process each color channel separately
#             t_b = np.maximum((self._A[0] - haze_img[:, :, 0].astype(float)) / (self._A[0] - self.C0),
#                              (haze_img[:, :, 0].astype(float) - self._A[0]) / (self.C1 - self._A[0]))
#             t_g = np.maximum((self._A[1] - haze_img[:, :, 1].astype(float)) / (self._A[1] - self.C0),
#                              (haze_img[:, :, 1].astype(float) - self._A[1]) / (self.C1 - self._A[1]))
#             t_r = np.maximum((self._A[2] - haze_img[:, :, 2].astype(float)) / (self._A[2] - self.C0),
#                              (haze_img[:, :, 2].astype(float) - self._A[2]) / (self.C1 - self._A[2]))
#         # Find the maximum value element-wise among t_b, t_g, and t_r
#             max_val = np.maximum(t_b, t_g, t_r)
#         # Apply boundary constraints: Limit Transmission to a maximum value of 1
#             self._Transmission = np.minimum(max_val, 1)
#         else:
#             # For grayscale images, perform a simplified version of the process
#             grayscale = np.maximum((self._A[0] - haze_img.astype(float)) / (self._A[0] - self.C0),
#                                             (haze_img.astype(float) - self._A[0]) / (self.C1 - self._A[0]))
#             self._Transmission = np.minimum(grayscale, 1)
#         # Apply morphological closing operation to further refine the Transmission map
#         kernel = np.ones((self.boundaryConstraint_windowSze, self.boundaryConstraint_windowSze), float)
#         self._Transmission = cv2.morphologyEx(self._Transmission, cv2.MORPH_CLOSE, kernel=kernel)

#     def load_filterbank(self):
#         kirsch_filters = []
#         kirsch_filters.append(np.array([[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]]))
#         kirsch_filters.append(np.array([[-3, -3, -3], [-3, 0, -3], [5, 5, 5]]))
#         kirsch_filters.append(np.array([[-3, -3, -3], [5, 0, -3], [5, 5, -3]]))
#         kirsch_filters.append(np.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]]))
#         kirsch_filters.append(np.array([[5, 5, -3], [5, 0, -3], [-3, -3, -3]]))
#         kirsch_filters.append(np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]]))
#         kirsch_filters.append(np.array([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]]))
#         kirsch_filters.append(np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]]))
#         kirsch_filters.append(np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]))
#         return kirsch_filters

#     def calculate_weighting_function(self, haze_img, _filter):

#         # Convert the hazy image to a double representation in the range [0, 1]
#         haze_image_double = haze_img.astype(float) / 255.0

#         # check if the image is color or gray scale.
#         if len(haze_img.shape == 3):

#             # For color image, extract the color channels (red, green, blue).
#             red = haze_image_double[:, :, 2]
#             d_r = self.circular_convolution_filter(red, _filter)

#             green = haze_image_double[:, :, 1]
#             d_g = self.circular_convolution_filter(green, _filter)

#             blue = haze_image_double[:, :, 0]
#             d_b = self.circular_convolution_filter(blue, _filter)

#             # calculate and return the weighting function based on color channels.
#             return np.exp(-((d_r ** 2) + (d_g ** 2) + (d_b ** 2)) / (2 * self.sigma * self.sigma))
#         else:
#             # for gray scale images, compute the weighting function based on single channel.
#             d = self.circular_convolution_filter(haze_image_double, _filter)

#             # calculate and return the weighting channel of gray scale images.
#             return np.exp(-((d ** 2) + (d ** 2) + (d ** 2)) / (2 * self.sigma * self.sigma))

#     def circular_convolution_filter(self, img, _filter):
#         # Get the dimensions of the filter
#         filter_height, filter_width = _filter.shape

#         # Ensure the filter is square and has an odd dimension
#         assert (filter_height == filter_width), 'Filter must be square in shape --> Height must be same as width'
#         assert (filter_height % 2 == 1), 'Filter dimension must be a odd number.'

#         # calculate half size of the filter.
#         filter_half_size = int((filter_height - 1) / 2)
#         rows, cols = img.shape

#         # pad the image with a circular wrap-around border.
#         padded_img = cv2.copyMakeBorder(img, filter_half_size, filter_half_size, filter_half_size, filter_half_size,
#                                         borderType=cv2.BORDER_WRAP)

#         # Perform 2D convolution on the padded image using the filter
#         filtered_img = cv2.filter2D(padded_img, -1, _filter)

#         # Extract the region of interest from the convolved image (remove padding)
#         result = filtered_img[filter_half_size:rows + filter_half_size, filter_half_size:cols + filter_half_size]
#         return result

#     def cal_transmission(self, haze_img):
#         # Get the dimensions of the transmission map
#         rows, cols = self._Transmission.shape
#         # Load and normalize Kirsch filters (used for edge detection)
#         kirsch_filters = self.load_filterbank()

#         # Normalize the filters
#         for idx, currentFilter in enumerate(kirsch_filters):
#             kirsch_filters[idx] = kirsch_filters[idx] / np.linalg.norm(currentFilter)

#         # Calculate Weighting function --> [rows, cols. numFilters] --> One Weighting function for every filter
#         WFun = []
#         for idx, currentFilter in enumerate(kirsch_filters):
#             WFun.append(self.calculate_weighting_function(haze_img, currentFilter))

#         # Precompute the constants that are later needed in the optimization step
#         tF = np.fft.fft2(self._Transmission)
#         DS = 0

#         for i in range(len(kirsch_filters)):
#             D = self.psf2otf(kirsch_filters[i], (rows, cols))
#             DS = DS + (abs(D) ** 2)

#         # Cyclic loop for refining t and u --> Section III in the paper
#         beta = 1  # Start Beta value --> selected from the paper
#         beta_max = 2 ** 4  # Selected from the paper --> Section III --> "Scene Transmission Estimation"
#         beta_rate = 2 * np.sqrt(2)  # Selected from the paper

#         while (beta < beta_max):
#             gamma = self.regularize_lambda / beta

#             # Fixing t first and solving for u
#             DU = 0
#             for i in range(len(kirsch_filters)):
#                 dt = self.circular_convolution_filter(self._Transmission, kirsch_filters[i])
#                 u = np.maximum((abs(dt) - (WFun[i] / (len(kirsch_filters) * beta))), 0) * np.sign(dt)
#                 DU = DU + np.fft.fft2(self.circular_convolution_filter(u, cv2.flip(kirsch_filters[i], -1)))

#             # Fixing u and solving t --> Equation 26 in the paper
#             # Note: In equation 26, the Numerator is the "DU" calculated in the above part of the code
#             # In the equation 26, the Denominator is the DS which was computed as a constant in the above code

#             self._Transmission = np.abs(np.fft.ifft2((gamma * tF + DU) / (gamma + DS)))
#             beta = beta * beta_rate

#         if self.showHazeTransmissionMap:
#             cv2.imshow("Haze Transmission Map", self._Transmission)
#             cv2.waitKey(1)

#     def removeHaze(self, haze_img):
#         ''' 
#         :param HazeImg: Hazy input image
#         :param Transmission: estimated transmission
#         :param A: estimated airlight
#         :param delta: fineTuning parameter for dehazing --> default = 0.85
#         :return: result --> Dehazed image
#         '''

#         epsilon = 0.0001
#         Transmission = pow(np.maximum(abs(self._Transmission), epsilon), self.delta)

#         haze_corrected_image = copy.deepcopy(haze_img)
#         if len(haze_img.shape == 3):
#             for ch in range(len(haze_img.shape)):
#                 temp = ((haze_img[:, :, ch].astype(float) - self._A[ch]) / Transmission) + self._A[ch]
#                 temp = np.maximum(np.minimum(temp, 255), 0)
#                 haze_corrected_image[:, :, ch] = temp
#         else:
#             temp = ((haze_img.astype(float) - self._A[0]) / Transmission) + self._A[0]
#             temp = np.maximum(np.minimum(temp, 255), 0)
#             haze_corrected_image = temp
#         return haze_corrected_image

#     def psf2otf(self, psf, shape):
#         if np.all(psf == 0):
#             return np.zeros_like(psf)

#         image_shape = psf.shape
#         # Pad the PSF to outsize
#         psf = self.zero_pad(psf, shape, position='corner')

#         # Circularly shift OTF so that the 'center' of the PSF is
#         # [0,0] element of the array
#         for axis, axis_size in enumerate(image_shape):
#             psf = np.roll(psf, -int(axis_size / 2), axis=axis)

#         # Compute the OTF
#         otf = np.fft.fft2(psf)

#         # Estimate the rough number of operations involved in the FFT
#         # and discard the PSF imaginary part if within roundoff error
#         # roundoff error  = machine epsilon = sys.float_info.epsilon
#         # or np.finfo().eps
#         n_ops = np.sum(psf.size * np.log2(psf.shape))
#         otf = np.real_if_close(otf, tol=n_ops)

#         return otf

#     def zero_pad(self, image, shape, position='corner'):
#         shape = np.asarray(shape, dtype=int)
#         image_shape = np.asarray(image.shape, dtype=int)

#         if np.alltrue(image_shape == shape):
#             return image

#         if np.any(shape <= 0):
#             raise ValueError("ZERO_PAD: null or negative shape given")

#         d_shape = shape - image_shape
#         if np.any(d_shape < 0):
#             raise ValueError("ZERO_PAD: target size smaller than source one")

#         pad_img = np.zeros(shape, dtype=image.dtype)

#         idx, idy = np.indices(image_shape)

#         if position == 'center':
#             if np.any(d_shape % 2 != 0):
#                 raise ValueError("ZERO_PAD: source and target shapes "
#                                  "have different parity.")
#             offx, offy = d_shape // 2
#         else:
#             offx, offy = (0, 0)

#         pad_img[idx + offx, idy + offy] = image

#         return pad_img

#     def remove_haze(self, foggy_image):
#         self.air_light_estimation(foggy_image)
#         self.boundary_constraint(foggy_image)
#         self.cal_transmission(foggy_image)
#         haze_corrected_img = self.removeHaze(foggy_image)
#         haze_transmission_map = self._Transmission
#         return haze_corrected_img, haze_transmission_map


# def remove_haze(foggy_image, airlightEstimation_windowSize=15, boundaryConstraint_windowSize=3, C0=20, C1=300,
#                 regularize_lambda=0.1, sigma=0.5, delta=0.85, showHazeTransmissionMap=True):
#     Dehazer = image_dehazer(airlightEstimation_windowSize=airlightEstimation_windowSize,
#                             boundaryConstraint_windowSize=boundaryConstraint_windowSize, C0=C0, C1=C1,
#                             regularize_lambda=regularize_lambda, sigma=sigma, delta=delta,
#                             showHazeTransmissionMap=showHazeTransmissionMap)
#     corrected_img, haze_transmission_map = Dehazer.remove_haze(foggy_image)
#     return corrected_img, haze_transmission_map





# importing libraries....
import cv2
import numpy as np
import copy


class image_dehazer():

    def __init__(self, airlightEstimation_windowSze=15, boundaryConstraint_windowSze=3, C0=20, C1=300,
                 regularize_lambda=0.1, sigma=0.5, delta=0.85, showHazeTransmissionMap=True):
        self.airlightEstimation_windowSze = airlightEstimation_windowSze
        self.boundaryConstraint_windowSze = boundaryConstraint_windowSze
        self.C0 = C0
        self.C1 = C1
        self.regularize_lambda = regularize_lambda
        self.sigma = sigma
        self.delta = delta
        self.showHazeTransmissionMap = showHazeTransmissionMap
        self._A = []
        self._transmission = []
        self._WFun = []

    def air_light_estimation(self, haze_img):
        # Check if the image is a color image or gray scale...
        if len(haze_img.shape == 3):
            # if it is a color image , process each color channel separately.
            for ch in range(len(haze_img.shape)):
              kernel = np.ones((self.airlightEstimation_windowSze, self.airlightEstimation_windowSze), np.uint8)
              min_img = cv2.erode(haze_img[:, :, ch], kernel)
                # Find the maximum value in image and append it to _A
              self._A.append(int(min_img.max()))
        else:
            # if it is a gray scale, use the same process without iterating through channel.
            kernel = np.ones((self.airlightEstimation_windowSze, self.airlightEstimation_windowSze), np.uint8)
            min_img = cv2.erode(haze_img, kernel)
            self._A.append(int(min_img.max()))

    def boundary_constraint(self, haze_img):
        # Check if the image is a color image or gray scale...
        if len(haze_img.shape == 3):
            # For color images, process each color channel separately
            t_b = np.maximum((self._A[0] - haze_img[:, :, 0].astype(float)) / (self._A[0] - self.C0),
                             (haze_img[:, :, 0].astype(float) - self._A[0]) / (self.C1 - self._A[0]))
            t_g = np.maximum((self._A[1] - haze_img[:, :, 1].astype(float)) / (self._A[1] - self.C0),
                             (haze_img[:, :, 1].astype(float) - self._A[1]) / (self.C1 - self._A[1]))
            t_r = np.maximum((self._A[2] - haze_img[:, :, 2].astype(float)) / (self._A[2] - self.C0),
                             (haze_img[:, :, 2].astype(float) - self._A[2]) / (self.C1 - self._A[2]))
        # Find the maximum value element-wise among t_b, t_g, and t_r
            max_val = np.maximum(t_b, t_g, t_r)
        # Apply boundary constraints: Limit Transmission to a maximum value of 1
            self._Transmission = np.minimum(max_val, 1)
        else:
            # For grayscale images, perform a simplified version of the process
            grayscale = np.maximum((self._A[0] - haze_img.astype(float)) / (self._A[0] - self.C0),
                                            (haze_img.astype(float) - self._A[0]) / (self.C1 - self._A[0]))
            self._Transmission = np.minimum(grayscale, 1)
        # Apply morphological closing operation to further refine the Transmission map
        kernel = np.ones((self.boundaryConstraint_windowSze, self.boundaryConstraint_windowSze), float)
        self._Transmission = cv2.morphologyEx(self._Transmission, cv2.MORPH_CLOSE, kernel=kernel)

    def load_filterbank(self):
        kirsch_filters = []
        kirsch_filters.append(np.array([[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]]))
        kirsch_filters.append(np.array([[-3, -3, -3], [-3, 0, -3], [5, 5, 5]]))
        kirsch_filters.append(np.array([[-3, -3, -3], [5, 0, -3], [5, 5, -3]]))
        kirsch_filters.append(np.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]]))
        kirsch_filters.append(np.array([[5, 5, -3], [5, 0, -3], [-3, -3, -3]]))
        kirsch_filters.append(np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]]))
        kirsch_filters.append(np.array([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]]))
        kirsch_filters.append(np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]]))
        kirsch_filters.append(np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]))
        return kirsch_filters

    def calculate_weighting_function(self, haze_img, _filter):

        # Convert the hazy image to a double representation in the range [0, 1]
        haze_image_double = haze_img.astype(float) / 255.0

        # check if the image is color or gray scale.
        if len(haze_img.shape == 3):

            # For color image, extract the color channels (red, green, blue).
            red = haze_image_double[:, :, 2]
            d_r = self.circular_convolution_filter(red, _filter)

            green = haze_image_double[:, :, 1]
            d_g = self.circular_convolution_filter(green, _filter)

            blue = haze_image_double[:, :, 0]
            d_b = self.circular_convolution_filter(blue, _filter)

            # calculate and return the weighting function based on color channels.
            return np.exp(-((d_r ** 2) + (d_g ** 2) + (d_b ** 2)) / (2 * self.sigma * self.sigma))
        else:
            # for gray scale images, compute the weighting function based on single channel.
            d = self.circular_convolution_filter(haze_image_double, _filter)

            # calculate and return the weighting channel of gray scale images.
            return np.exp(-((d ** 2) + (d ** 2) + (d ** 2)) / (2 * self.sigma * self.sigma))

    def circular_convolution_filter(self, img, _filter):
        # Get the dimensions of the filter
        filter_height, filter_width = _filter.shape

        # Ensure the filter is square and has an odd dimension
        assert (filter_height == filter_width), 'Filter must be square in shape --> Height must be same as width'
        assert (filter_height % 2 == 1), 'Filter dimension must be a odd number.'

        # calculate half size of the filter.
        filter_half_size = int((filter_height - 1) / 2)
        rows, cols = img.shape

        # pad the image with a circular wrap-around border.
        padded_img = cv2.copyMakeBorder(img, filter_half_size, filter_half_size, filter_half_size, filter_half_size,
                                        borderType=cv2.BORDER_WRAP)

        # Perform 2D convolution on the padded image using the filter
        filtered_img = cv2.filter2D(padded_img, -1, _filter)

        # Extract the region of interest from the convolved image (remove padding)
        result = filtered_img[filter_half_size:rows + filter_half_size, filter_half_size:cols + filter_half_size]
        return result

    def cal_transmission(self, haze_img):
        # Get the dimensions of the transmission map
        rows, cols = self._Transmission.shape
        # Load and normalize Kirsch filters (used for edge detection)
        kirsch_filters = self.load_filterbank()

        # Normalize the filters
        for idx, currentFilter in enumerate(kirsch_filters):
            kirsch_filters[idx] = kirsch_filters[idx] / np.linalg.norm(currentFilter)

        # Calculate Weighting function --> [rows, cols. numFilters] --> One Weighting function for every filter
        WFun = []
        for idx, currentFilter in enumerate(kirsch_filters):
            WFun.append(self.calculate_weighting_function(haze_img, currentFilter))

        # Precompute the constants that are later needed in the optimization step
        tF = np.fft.fft2(self._Transmission)
        DS = 0

        for i in range(len(kirsch_filters)):
            D = self.psf2otf(kirsch_filters[i], (rows, cols))
            DS = DS + np.multiply(np.conj(D), D)
            WD = np.fft.fft2(np.multiply(WFun[i], self._Transmission))
            tF = tF + self.delta * np.conj(D) * WD

        # Store the computed weighting functions for later use
        self._WFun = WFun

        # compute inverse FFT to obtain the transmission map in the spatial domain.
        self._Transmission = np.abs(np.fft.ifft2(np.divide(tF, (1 + self.delta * DS))))
        if self.showHazeTransmissionMap:
            cv2.imshow('Haze Transmission Map', self._Transmission)

    def remove_haze(self, haze_img):

        # Estimate air-light (global atmospheric light) from the hazy image
        self.air_light_estimation(haze_img)
        # Apply boundary constraints to estimate the initial transmission map
        self.boundary_constraint(haze_img)
        # Refine the transmission map using edge-preserving regularization
        self.cal_transmission(haze_img)

        # Calculate the number of channels in the input image
        if len(haze_img.shape) == 3:
            num_channels = haze_img.shape[2]
        else:
            num_channels = 1

        # Apply the dehazing formula to recover the haze-free image
        HazeCorrectedImg = np.zeros(haze_img.shape)
        for ch in range(num_channels):
            HazeCorrectedImg[:, :, ch] = self._A[ch] + (haze_img[:, :, ch] - self._A[ch]) / self._Transmission

        # Convert the dehazed image to an 8-bit unsigned integer representation
        return HazeCorrectedImg.astype(np.uint8), self._Transmission

    def psf2otf(self, psf, shape):
        # Convert point spread function to optical transfer function
        psf_size = np.array(psf.shape)
        pad_size = shape - psf_size
        # Pad the PSF to the desired shape
        psf = self.zero_pad(psf, shape, position='corner')

        # Circularly shift the PSF to the center of the image
        for axis, axis_size in enumerate(psf_size):
            psf = np.roll(psf, -int(axis_size / 2), axis=axis)

        # Compute the Fourier transform of the shifted PSF
        otf = np.fft.fft2(psf)

        # Set a tolerance value to consider small values as zero
        n_ops = np.sum(psf_size * np.log2(psf_size))
        otf[np.abs(otf) < n_ops * np.finfo(float).eps] = 0

        return otf

    def zero_pad(self, image, shape, position='corner'):
        # Determine the size of the original image
        imshape = np.array(image.shape)
        # Check if the desired shape matches the image shape, and if so, return the original image
        if np.all(imshape == shape):
            return image

        # Create a zero-padded image with the desired shape
        pad_img = np.zeros(shape, dtype=image.dtype)

        # Position the original image in the zero-padded array based on the specified position
        if position == 'corner':
            pad_img[:imshape[0], :imshape[1]] = image
        elif position == 'center':
            # Calculate the center coordinates for positioning the original image
            start = (shape - imshape) // 2
            pad_img[start[0]:start[0] + imshape[0], start[1]:start[1] + imshape[1]] = image

        return pad_img


