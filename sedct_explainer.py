import warnings
warnings.simplefilter("always", UserWarning)
warnings.simplefilter("once", category=UserWarning)

import numbers 
from typing import Optional, Union
from image_segmentations import *
from model_wrappers import *





class SEDCTExplainer(): 
    def __init__(self, classifier: Optional[BaseModelWrapper] = None,
                 image_segmentation: Optional[BaseImageSegmentation] = None) -> None : 
        
        self._counterfactual_id = None

        if classifier is not None:
            self._set_classifier(classifier)
        else: 
            self.classifier = None

        if image_segmentation is not None:
            self._set_image_segmentation(image_segmentation)
        else:
            self.image_segmentation = None

    
    def _set_classifier(self, classifier) -> None :
        # check if classifier is an object of class BaseModelWrapper
        if not isinstance(classifier, BaseModelWrapper):
            raise TypeError(f"The algorithm support only object of type BaseModelWrapper. "
                            f"The provided object type ({type(classifier)}) is incorrect.")
        self.classifier = classifier


    def _set_image_segmentation(self, image_segmentation) -> None :
        # check if image_segmentation is an object of class BaseImageSegmentation
        if not isinstance(image_segmentation, BaseImageSegmentation):
            raise TypeError(f"The algorithm support only object of type BaseImageSegmentation. "
                            f"The provided object type ({type(image_segmentation)}) is incorrect.")
        self.image_segmentation = image_segmentation


    def show_counterfactual(self) -> None :
        '''Show the counterfactual obtained.'''
        if self._counterfactual_id is None:
            raise ValueError("There is no counterfactual. Please run the 'explain_instance' function first.")
        
        import matplotlib.pyplot as plt
        img = self.get_counterfactual()
        if isinstance(img, Tensor):
            if len(img.shape) == 3:
                if img.shape[0] == 1:
                    img = img.squeeze()
                    plt.imshow(img, cmap='gray')
                else:
                    plt.imshow(img.permute(1,2,0))
            elif len(img.shape) == 2:
                plt.imshow(img.numpy(), cmap='gray')
            else:
                raise ValueError("Unsupported tensor shape for image display.")
        else: 
            if len(img.shape) == 3:
                plt.imshow(img)
            elif len(img.shape) == 2:
                plt.imshow(img, cmap='gray')
            else:
                raise ValueError("Unsupported tensor shape for image display.")
        plt.axis('off')
        plt.show()


    def explain_instance(
            self, 
            classifier: Optional[BaseModelWrapper] = None,
            image_segmentation: Optional[BaseImageSegmentation] = None, 
            target: Optional[int] = None, 
            max_iter: Optional[Union[int, float]] = 10, 
            show: Optional[bool] = False
            ) -> None : 
        import numpy as np
        
        # Check classifier
        if classifier is not None:
            self._set_classifier(classifier)
        else:
            if self.classifier is None: 
                raise TypeError('self.classifier cannot be None. Please specify a model to explain.')
        assert self.classifier is not None

        # Check image_segmentation
        if image_segmentation is not None:
            self._set_image_segmentation(image_segmentation)
        else:
            if self.image_segmentation is None:
                raise TypeError('self.image_segmentation cannot be None. ' \
                                'Please define an object of class TorchImageSegmentation or SKLearnImageSegmentation.')
        assert self.image_segmentation is not None

        # Check target REDO 
        if target is None:
            # Check if the model is multiclass
            if self.classifier.is_multiclass():
                raise ValueError('You have to specify a target class to explain correctly the given multiclass classifier.')
            else:
                t = None
        elif isinstance(target, numbers.Integral):
            # Check target positive
            if target < 0:
                raise ValueError(f"'target' parameter should be a positive integer. "
                                 f"A negative integer ({target}) has been provided. Please provide a valid number.")
            # Check target in range of classes 
            if target not in range(self.classifier.get_n_classes()):
                raise IndexError(f"The class specified is outiside the classes range. "
                                 f"Number of classes: {self.classifier.get_n_classes()}, target class specified: {target}")
            t = target
        else:
            raise TypeError(f'target parameter must be an integer number or None. Provided target type: {type(target)}')

        # Check show
        if not isinstance(show, bool):
            raise TypeError(f'show parameter must be of type bool. Provided type ({type(show)}) is incorrect.')


        image = self.image_segmentation.get_image()
        n_segments = self.image_segmentation.get_n_segments()
        segments_ids = np.unique(self.image_segmentation.get_segments()).tolist()

        # Check max_iter
        if not isinstance(max_iter, (numbers.Integral, numbers.Real)):
            raise TypeError(f'max_iter parameter must be a positive integer or float. Provided max_iter type: {type(max_iter)}')
        else:
            if max_iter < 0: 
                if max_iter == -1:
                    max_iter = n_segments
                else:
                    raise ValueError(f"'max_iter' should be a positive number or -1. A negative number has been provided: {max_iter}.")
            elif max_iter > 0 and max_iter < 1:
                max_iter = int(n_segments * max_iter)
            else:
                if isinstance(max_iter, numbers.Real):
                    warnings.warn(f"'max_iter' is a float outside [0,1]: it has been converted to {int(max_iter)}.")
                max_iter = int(max_iter)


        c = self.classifier.get_pred(image)
        p_c = self.classifier.get_score(image)

        R = []
        P_R = []
        C = []
        P = []

        # Start of algo
        for id in segments_ids:
            new_image = self.image_segmentation.remove_segment(id)
            c_new = self.classifier.get_pred(new_image)

            if t is None:
                # Binary classification
                p_c_new = self.classifier.get_score(new_image)
                if c_new != c:
                    R.append(id)
                    P_R.append(p_c_new) 
                else:
                    C.append(id)
                    P.append(p_c - p_c_new)
            else:
                # Multiclass classification
                p_c_new, p_t_new = self.classifier.get_score(new_image, target=t)
                if c_new == t:
                    R.append(id)
                    P_R.append(p_c_new) 
                else:
                    C.append(id)
                    P.append(p_t_new - p_c_new)

        current_iter = 0
        while not R:
            k = np.argmax(P)
            best = C[k]
            if isinstance(best, numbers.Integral):
                best = [best]
            best_set = []
            for id in segments_ids:
                temp = best.copy()
                temp.append(id)
                temp = list(set(temp))
                best_set.append(temp)

            C.pop(k)
            P.pop(k)
            for id_list in best_set:
                new_image = self.image_segmentation.remove_segment(id_list)
                c_new = self.classifier.get_pred(new_image)

                if t is None:
                    # Binary classification
                    p_c_new = self.classifier.get_score(new_image)
                    if c_new != c:
                        R.append(id_list)
                        P_R.append(p_c_new) 
                    else:
                        C.append(id_list)
                        P.append(p_c - p_c_new)
                else:
                    # Multiclass classification
                    p_c_new, p_t_new = self.classifier.get_score(new_image, target=t)

                    if c_new == t:
                        R.append(id_list)
                        P_R.append(p_c_new) 
                    else:
                        C.append(id_list)
                        P.append(p_t_new - p_c_new)
            
            # max_iter controll
            current_iter += 1
            if current_iter > max_iter:
                break 
            
        if R:
            max = np.argmax(list(P_R))
            R = R[max]
            self._counterfactual_id = R
        else: 
            print(f"The algorithm has reached 'max_iter' value ({max_iter}). No counterfactual has been found. "
                f"To produce a counterfactual try to increse the 'max_iter' parameter.")
            
        if show: 
            if self._counterfactual_id is None:
                print('The algorithm has not found any counterfactual. No image can be produced.')
            else: 
                self.show_counterfactual()



    def get_counterfactual(self) -> ndarray | Tensor : 
        # return an image
        if self.image_segmentation is None:
            raise ValueError("'self.image_segmentation' is None. Please provide a segmentation object before.")
        if self._counterfactual_id is None:
            raise ValueError("There is no counterfactual. Please run the 'explain_instance' method before.")

        return self.image_segmentation.get_ids_image(self._counterfactual_id) 
    


    def get_counterfactual_mask(self) -> ndarray:
        # return a binary mask of the image
        if self._counterfactual_id is None:
            raise ValueError("Counterfactuals are not present. Please run the 'explain_instance' first.")
        if self.image_segmentation is None:
            raise ValueError("self.image_segmentation parameter is None. Please provide a valid parameter first.")
        else: 
            return self.image_segmentation.get_ids_mask(self._counterfactual_id)

