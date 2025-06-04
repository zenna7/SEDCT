from abc import ABC, abstractmethod
from typing import Union, Optional
from torch import Tensor
from numpy import ndarray
from PIL.Image import Image
import numbers
import warnings
warnings.simplefilter("always", UserWarning)
warnings.simplefilter("once", category=UserWarning)






class BaseImageSegmentation(ABC):
    def __init__(self, image: Union[Tensor, ndarray, Image, list],
                 segmentation_method: Optional[str] = None,
                 fill_method: Optional[str] = None,
                 **kwargs) -> None :
        """
        Args:
            image (Tensor, ndarray, Image, list): The image object to explain.
        """
        self.image = None
        self._set_image(image)

        if segmentation_method is not None:
            self._set_segmentation_method(segmentation_method)
        else:
            self.segmentation_method = None
        
        if fill_method is not None:
            self._set_fill_method(fill_method)
        else:
            self.fill_method = None
        
        self.options = kwargs
        self._segments = None
        self._n_segments = None


    @abstractmethod
    def _set_image(self, image) -> None: 
        '''Check and set the internal image to the right format.'''
        pass

    @abstractmethod
    def compute_segmentation(self, segmentation_method: Optional[str] = None, 
                             show: bool = False,
                             **kwargs) -> None:
        '''Run the segmentation algorithm.'''
        pass

    @abstractmethod 
    def remove_segment(self, segment: Union[int, list[int]], 
                       fill_method: Optional[str] = None) -> ndarray | Tensor :
        '''Remove the given segments from the image.'''
        pass

    @abstractmethod
    def get_segmented_image(self, **kwargs) -> ndarray | Tensor:
        '''Return the image segmented.'''
        pass
    
    @abstractmethod
    def show_image(self) -> None :
        '''Plot the plain image.'''
        pass
    
    @abstractmethod
    def show_segmented_image(self, **kwargs) -> None :
        '''Plot the segmented image.'''
        pass

    @abstractmethod
    def get_ids_image(self, id: Union[int, list[int]]) -> ndarray | Tensor :
        pass


    def get_ids_mask(self, id: Union[int, list[int]]) -> ndarray :
        import numpy as np
        if isinstance(id, numbers.Integral):
            ids = [id]
        elif isinstance(id, list): 
            if not all(isinstance(x, numbers.Integral) for x in id):
                raise TypeError("All elements in 'id' list must be integers. An invalid element has been provided.")
            ids = id
        else:
            raise TypeError(f"'id' parameter must be of type int or list of int. Provided type ({type(id)}) is invalid.")
        
        if self._segments is None:
            raise ValueError("Segmentation not yet computed. Run 'compute_segmentation' first.")
        
        mask = np.isin(self._segments, ids).astype(np.uint8)
        return mask


    def get_image(self) -> ndarray | Tensor :
        '''Return the plain image.'''
        if self.image is None:
            raise FileNotFoundError('No image have been precedently provided.')
        else: 
            return self.image


    def get_segments(self) -> ndarray :
        '''Return the _segments attribute'''
        if self._segments is None:
            raise ValueError("self._segments is none. Please run the 'compute_segmentation' function first.")
        else:
            return self._segments


    def get_n_segments(self) -> int:
        if self._n_segments is None:
            if self.segmentation_method is not None and self.fill_method is not None:
                self.compute_segmentation()
            else:
                raise ValueError("Please run the 'compute_segmentation' function first to " \
                                 "obtain the number of segments.")
        assert self._n_segments is not None
        return self._n_segments


    def _image_to_ndarray(self, image) -> ndarray:
        '''Convert and return any image in a Numpy ndarray.'''
        import numpy as np

        # Handle PIL Image and lists
        if isinstance(image, (Image, list)):
            img = np.array(image)
        # Handle torch tensor
        elif isinstance(image, Tensor):
            if image.dim() == 3:
                img = image.permute(1, 2, 0).detach().cpu().numpy()
            elif image.dim() == 2:
                img = image.detach().cpu().numpy()
            else:
                raise ValueError(f"Unsupported tensor shape: {image.shape}")
        # Handle NumPy array
        elif isinstance(image, np.ndarray):
            img = image
        else:
            raise TypeError(f"Unsupported image type: {type(image)}. Please provide an image in one of the following formats: PIL Image, list, Numpy array, Torch tensor")

        # Reject batched input
        if img.ndim == 4:
            raise ValueError(f"Batched images are not supported. Got shape: {img.shape}")
        # Handle RGBA by dropping alpha
        if img.ndim == 3 and img.shape[2] == 4:
            warnings.warn("RGBA image detected. Dropping alpha channel.")
            img = img[:, :, :3]
        # Handle (H, W, 1)
        if img.ndim == 3 and img.shape[2] == 1:
            img = img.squeeze(-1)
        if img.ndim == 2:
            pass
        elif img.ndim == 3:
            if img.shape[2] not in [1, 3]:
                raise ValueError(f"Unsupported number of channels: {img.shape[2]}")
        else:
            raise ValueError(f"Unsupported image dimensions: {img.ndim}")
        
        return img


    def _set_segmentation_method(self, method) -> None :
        #function that check the support for the given segmentation method
        if not isinstance(method, str):
            raise TypeError(f'segmentation_method must be a string. Provided fill_color type ({type(method)}) is incorrect.')
        
        supported = ['slic', 'felzenszwalb', 'quickshift', 'chan_vese', 'morphological_chan_vese', 
                     'watershed', 'random_walker', 'morphological_geodesic_active_contour']
        method = method.lower()
        if method not in supported:
            raise NotImplementedError(f"segmentation_method provided is not supported. "
                                      f"Supported segmentation_method values are: {', '.join(supported)}.")

        self.segmentation_method = method

    
    def _set_fill_method(self, method) -> None :
        # check that the fill_method is a string and a method supported
        if not isinstance(method, str):
            raise TypeError(f'fill_method must be a string. Provided fill_method type ({type(method)}) is incorrect.')
        
        supported = ['mean', 'fill', 'fill_white', 
                     'fill_gray', 'fill_red', 'fill_green', 'fill_blue']
        if method not in supported:
            raise NotImplementedError(f"fill_method provided is not supported. "
                                      f"Supported fill_method values are: {', '.join(supported)}.")
        
        self.fill_method = method


    def _set_n_segments(self) -> None :
        '''Compute and set the total number of segments'''
        if self._segments is not None:
            import numpy as np
            self._n_segments = len(np.unique(self._segments))
        else:
            raise SystemError('self.segments has not been correctly computed. Currently it is None')


    def _skimage_segmentation(self, image, **kwargs) -> None:
        from skimage.segmentation import (slic, felzenszwalb, quickshift, chan_vese, morphological_chan_vese, 
                                          watershed, random_walker, morphological_geodesic_active_contour)
        from skimage.filters import sobel
        from skimage.color import rgb2gray

        if self.segmentation_method is None:
            warnings.warn("No segmentation method was specified. 'slic' method with default parameters as been used.", category=UserWarning)
            self.segmentation_method = 'slic'
            if self.image.ndim == 2:
                self.options["channel_axis"] = None

        options = {**self.options, **kwargs}

        match self.segmentation_method:
            case 'slic':
                self._segments = slic(image, **options)
            
            case 'felzenszwalb':
                self._segments = felzenszwalb(image, **options)
            
            case 'quickshift':
                self._segments = quickshift(image, **options)
            
            case 'chan_vese':
                self._segments = chan_vese(image, **options)
            
            case 'morphological_chan_vese':
                self._segments = morphological_chan_vese(image, **options)
            
            case 'watershed':
                gradient = options.pop('gradient', sobel(rgb2gray(image)))
                self._segments = watershed(gradient, **options)
            
            case 'random_walker':
                if 'markers' not in options:
                    raise ValueError("'random_walker' segmentation requires 'markers' in kwargs.")
                self._segments = random_walker(image, **options)

            case 'morphological_geodesic_active_contour':
                self._segments = morphological_geodesic_active_contour(image, **options) 

        self._set_n_segments()


    def _remove_segment(self, image, segment) -> ndarray:
        import numpy as np
        output_image = image.copy()

        is_float = image.max() <= 1.0
        max_val = 1.0 if is_float else 255
        zero_val = 0.0 if is_float else 0

        if len(self.image.shape) == 2: 
            match self.fill_method:
                case 'mean':
                    for seg in segment:
                        mask = np.isin(self._segments, seg)
                        mean_val = image[mask].mean()
                        output_image[mask] = mean_val
                case 'fill':
                    mask = np.isin(self._segments, segment)
                    output_image[mask] = zero_val
                case 'fill_white':
                    mask = np.isin(self._segments, segment)
                    output_image[mask] = max_val
                case _:
                    raise SystemError(f"The methods: ['fill_red', 'fill_blue', 'fill_green', 'fill_gray'] are not supported for grayscale images.")
        
        elif len(self.image.shape) == 3:
            match self.fill_method:
                case 'mean':
                    for seg in segment:
                        mask = np.isin(self._segments, seg)
                        for c in range(3):
                            mean_val = image[mask, c].mean()
                            output_image[mask, c] = mean_val
                case 'fill':
                    mask = np.isin(self._segments, segment)
                    output_image[mask] = [zero_val] * 3
                case 'fill_white':
                    mask = np.isin(self._segments, segment)
                    for c in range(3):
                        output_image[mask, c] = max_val
                case 'fill_red':
                    mask = np.isin(self._segments, segment)
                    output_image[mask] = [max_val, zero_val, zero_val]
                case 'fill_green':
                    mask = np.isin(self._segments, segment)
                    output_image[mask] = [zero_val, max_val, zero_val]
                case 'fill_blue':
                    mask = np.isin(self._segments, segment)
                    output_image[mask] = [zero_val, zero_val, max_val]
                case 'fill_gray':
                    mask = np.isin(self._segments, segment)
                    gray_val = max_val * 0.5  
                    output_image[mask] = [gray_val] * 3
        else:
            raise SystemError('Unsupported image shape.')

        return output_image








class TorchImageSegmentation(BaseImageSegmentation):
    def __init__(self, image: Union[Tensor, ndarray, Image, list],
                 segmentation_method: Optional[str] = None,
                 fill_method: Optional[str] = None,
                 **kwargs) -> None :
        super().__init__(image, segmentation_method, fill_method, **kwargs)



    def _tensor_to_ndarray(self, img_tensor) -> ndarray :
        '''Converts a PyTorch tensor to a NumPy ndarray.'''
        if img_tensor.requires_grad:
            img_tensor = img_tensor.detach()
        img_tensor = img_tensor.cpu()

        if img_tensor.dim() == 3:
            return img_tensor.permute(1, 2, 0).numpy()
        elif img_tensor.dim() == 2:
            return img_tensor.numpy()
        else:
            raise ValueError(f"Unsupported tensor shape: {img_tensor.shape}")



    def _ndarray_to_tensor(self, img_ndarray) -> Tensor : 
        '''Converts a NumPy ndarray to a PyTorch tensor.'''
        import torch
        image = img_ndarray.copy()
        if image.ndim == 3:
            return torch.from_numpy(image).permute(2, 0, 1)  
        elif image.ndim == 2:
            return torch.from_numpy(image)
        else:
            raise ValueError(f"Unsupported ndarray shape: {img_ndarray.shape}")



    def _set_image(self, image) -> None :
        '''Set the image to the right format.'''
        if not isinstance(image, (Tensor, ndarray, Image, list)):
            raise TypeError("Invalid image type. Image must be of type 'torch.Tensor', 'numpy.ndarray', 'PIL.Image.Image', list")
        else:
            image_ndarray = self._image_to_ndarray(image)
            self.image = self._ndarray_to_tensor(image_ndarray)



    def compute_segmentation(self, segmentation_method: Optional[str] = None, 
                             show: bool = False,
                             **kwargs) -> None:
        # if parameters are passed compute the segmentation using them,
        # else check the presence of parameters in self
        if segmentation_method is not None:
            if self.image is None:
                raise FileNotFoundError('No image have been provided.')
            else:
                self._set_segmentation_method(segmentation_method)
        else:
            if self.image is None:
                raise FileNotFoundError('No image have been provided.')

        if not isinstance(show, bool):
            raise TypeError(f'show parameter must be of type bool. Provided type ({type(show)}) is incorrect')
        
        if kwargs:
            self.options = {**self.options, **kwargs}
        
        image = self.image.clone()
        image_ndarray = self._tensor_to_ndarray(image)
        self._skimage_segmentation(image_ndarray, **kwargs)

        if show:
            self.show_segmented_image()


    def show_image(self) -> None :
        '''Plot the plain image.'''
        if self.image is None:
            raise ValueError("No image has been provided.")
        
        import matplotlib.pyplot as plt
        plt.imshow(self.image.permute(1,2,0))
        plt.axis('off')


    def show_segmented_image(self, **kwargs) -> None :
        '''Plot the segmented image.'''
        if self.image is None:
            raise ValueError("No image has been provided.")
        if self._segments is None:
            raise ValueError("No segmentation has been computed. Please run the 'compute_segmentation' method first.")
        
        import matplotlib.pyplot as plt
        from skimage.segmentation import mark_boundaries
        img = self._tensor_to_ndarray(self.image)
        segmented_image = mark_boundaries(img, self._segments, **kwargs)
        plt.imshow(segmented_image)
        plt.axis('off')


    def get_segmented_image(self, **kwargs) -> Tensor:
        '''Return the image segmented.'''
        if self.image is None:
            raise ValueError("No image has been provided.")
        if self._segments is None:
            raise ValueError("No segmentation has been computed. Please run the 'compute_segmentation' method first.")
                
        from skimage.segmentation import mark_boundaries
        img = self._tensor_to_ndarray(self.image)
        segmented_image = mark_boundaries(img, self._segments, **kwargs)
        return self._ndarray_to_tensor(segmented_image)
        

    def remove_segment(self, segment: Union[int, list[int]], 
                       fill_method: Optional[str] = None) -> Tensor: 
        
        if not isinstance(segment, (numbers.Integral, list)):
            raise TypeError(f"segment parameter supports only integers or list of integers. "
                            f"The provided segment type ({type(segment)}) is invalid.")
        
        if isinstance(segment, list):
            if not all(isinstance(x, numbers.Integral) for x in segment):
                raise TypeError('segment parameter supports only integers or list of integers. The provided list present an invalid type.')
        else:
            segment = [segment]

        if self._n_segments is not None:
            if not all(x in range(self._n_segments) for x in segment):
                raise IndexError(f"One or more segment indexes provided are out of range _n_segments ({self._n_segments})")
        else:
            raise SystemError("self._n_segments parameter is 'None', compute_segmentation function probably didn't run properly.")

        # check the presence of an image 
        if self.image is None:
            raise FileNotFoundError('No image have been precedently provided.')
        
        #pass new parameters
        if fill_method is not None:
            self._set_fill_method(fill_method)
        else:
            self.fill_method = 'fill'
            warnings.warn("No method to fill the removed segment(s) was specified. 'fill' method as been used as default.", category=UserWarning)


        #compute segment removal
        image = self.image.clone()
        img_ndarray = self._tensor_to_ndarray(image)
        output_ndarray = self._remove_segment(img_ndarray, segment)
        return self._ndarray_to_tensor(output_ndarray)


    def get_ids_image(self, id: Union[int, list[int]]) -> Tensor:
        if self.image is None:
            raise ValueError('No image has been provided. Please provide an image.')
        
        import numpy as np
        image = self.image.clone()
        img_ndarray = self._tensor_to_ndarray(image)
        mask = self.get_ids_mask(id)
        if mask.ndim == 2 and img_ndarray.ndim == 3:
            mask = mask[:, :, np.newaxis]
        masked_image = img_ndarray * mask

        return self._ndarray_to_tensor(masked_image)








class SKimageImageSegmentation(BaseImageSegmentation):
    def __init__(self, image: Union[Tensor, ndarray, Image, list],
                 segmentation_method: Optional[str] = None,
                 fill_method: Optional[str] = None,
                 **kwargs) -> None :
        super().__init__(image, segmentation_method, fill_method, **kwargs)
        

        
    def _set_image(self, image) -> None :
        '''Set the image to the right format.'''
        if not isinstance(image, (Tensor, ndarray, Image, list)):
            raise TypeError("Invalid image type. Image must be of type 'torch.Tensor', 'numpy.ndarray', 'PIL.Image.Image', list")
        else:
            self.image = self._image_to_ndarray(image)



    def compute_segmentation(self, segmentation_method: Optional[str] = None, 
                             show: bool = False,
                             **kwargs) -> None:
        # if parameters are passed compute the segmentation using them,
        # else check the presence of parameters in self
        if segmentation_method is not None:
            if self.image is None:
                raise FileNotFoundError('No image have been provided.')
            else:
                self._set_segmentation_method(segmentation_method)
        else:
            if self.image is None:
                raise FileNotFoundError('No image have been provided.')

        if not isinstance(show, bool):
            raise TypeError(f'show parameter must be of type bool. Provided type ({type(show)}) is incorrect')
        
        if kwargs:
            self.options = {**self.options, **kwargs}
        
        image = self.image.copy()
        self._skimage_segmentation(image, **kwargs)

        if show:
            self.show_segmented_image()



    def remove_segment(self, segment: Union[int, list[int]], 
                       fill_method: Optional[str] = None) -> ndarray: 
        
        if not isinstance(segment, (numbers.Integral, list)):
            raise TypeError(f"segment parameter supports only integers or list of integers. "
                            f"The provided segment type ({type(segment)}) is invalid.")
        
        if isinstance(segment, list):
            if not all(isinstance(x, numbers.Integral) for x in segment):
                raise TypeError('segment parameter supports only integers or list of integers. The provided list present an invalid type.')
        else:
            segment = [segment]

        if self._n_segments is not None:
            if not all(x in range(self._n_segments) for x in segment):
                raise IndexError(f"One or more segment indexes provided are out of range _n_segments ({self._n_segments})")
        else:
            raise SystemError("self._n_segments parameter is 'None', compute_segmentation function probably didn't run properly.")

        # check the presence of an image 
        if self.image is None:
            raise FileNotFoundError('No image have been precedently provided.')
        
        # Pass new parameters
        if fill_method is not None:
            self._set_fill_method(fill_method)
        else:
            self.fill_method = 'fill'
            warnings.warn("No method to fill the removed segment(s) was specified. 'fill' method as been used as default.", category=UserWarning)

        # Compute segment removal
        image = self.image.copy()
        return self._remove_segment(image, segment)



    def get_ids_image(self, id: Union[int, list[int]]) -> ndarray:
        if self.image is None:
            raise ValueError('No image has been provided. Please provide an image.')
        
        import numpy as np
        image = self.image.copy()
        mask = self.get_ids_mask(id)
        if mask.ndim == 2 and image.ndim == 3:
            mask = mask[:, :, np.newaxis]
        masked_image = image * mask

        return masked_image



    def show_image(self) -> None :
        '''Plot the plain image.'''
        if self.image is None:
            raise ValueError("No image has been provided.")
        import matplotlib.pyplot as plt
        plt.imshow(self.image)
        plt.axis('off')


    def show_segmented_image(self, **kwargs) -> None :
        '''Plot the segmented image.'''
        if self.image is None:
            raise ValueError("No image has been provided.")
        if self._segments is None:
            raise ValueError("No segmentation has been computed. Please run the 'compute_segmentation' method first.")
        
        import matplotlib.pyplot as plt
        from skimage.segmentation import mark_boundaries
        segmented_image = mark_boundaries(self.image, self._segments, **kwargs)
        plt.imshow(segmented_image)
        plt.axis('off')


    def get_segmented_image(self, **kwargs) -> ndarray:
        '''Return the image segmented.'''
        if self.image is None:
            raise ValueError("No image has been provided.")
        if self._segments is None:
            raise ValueError("No segmentation has been computed. Please run the 'compute_segmentation' method first.")
                
        from skimage.segmentation import mark_boundaries
        return mark_boundaries(self.image, self._segments, **kwargs)



# TODO list: 
#  comment the code correctly
#  final checks of types and typos