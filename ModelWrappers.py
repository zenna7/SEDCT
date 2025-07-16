from abc import ABC, abstractmethod
from typing import Any, Optional, Union, Callable

from torch import Tensor
import torch.nn as nn

import numbers
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from numpy import ndarray

import warnings
warnings.simplefilter("always", UserWarning)
warnings.simplefilter("once", category=UserWarning)





class BaseModelWrapper(ABC): 
    def __init__(self):
        self._n_classes = None

    @abstractmethod
    def _set_model(self, model) -> None :
        '''Check and set the model.'''
        pass
    
    @abstractmethod
    def get_pred(self, image) -> int:
        # function to get the class prediction give the image 
        pass

    @abstractmethod
    def get_score(self, image, target: Optional[int] = None) -> Union[float, tuple[float, float]] :
        # function to get the score of the prediction of the given image
        pass


    def is_multiclass(self) -> bool:
        '''Boolean function that return 'True' if the model is multiclass, 'False' otherwise.'''
        if self._n_classes is None:
            raise ValueError('No model has been provided. Please provide a model before.')
        else:
            return self._n_classes > 1


    def get_n_classes(self) -> int:
        '''Return the number of classes the model can classify.'''
        if self._n_classes is None:
            raise ValueError('No model has been provided. Please provide a model before.')
        else:
            return self._n_classes 





class SKLearnModelWrapper(BaseModelWrapper):
    def __init__(self, model: BaseEstimator, resize_shape: Optional[tuple] = None, flatten: bool = False, **kwargs) -> None:
        super().__init__()
        """
        Args:
            model: A fitted scikit-learn classifier or compatible wrapper.
            Supported types:
            - sklearn classifiers (e.g., LogisticRegression, RandomForestClassifier)
            - sklearn Pipelines ending in a classifier
            - GridSearchCV / RandomizedSearchCV
            - Ensemble classifiers (VotingClassifier, StackingClassifier, etc.)
            - XGBClassifier, LGBMClassifier

            resize: .....

            The model must be fitted and expose either `classes_` or allow inference of class count.
        """
        # Allow for parameters passed in the init method for advanced usage
        n_classes = kwargs.get("n_classes", None)

        if n_classes is not None:
            if not isinstance(n_classes, numbers.Integral):
                raise ValueError(f"n_classes parameter should be of type int. Invalid type ({type(n_classes)}) has been provided.")
            else:
                self._n_classes = n_classes

        resize_method = kwargs.get("resize_method", None)
        if resize_method is not None:
            if not isinstance(resize_method, str):
                raise ValueError(f"resize_method parameter should be of type string. Invalid type ({type(resize_method)}) has been provided.")
            else:
                self.resize_method = resize_method
        else: 
            self.resize_method = resize_method

        if resize_shape is not None:
            if not isinstance(resize_shape, tuple):
                raise ValueError(f"resize_shape parameter should be of type float. Invalid type({type(resize_shape)}) has been provided.")
            elif len(resize_shape) != 2:
                raise ValueError(f"resize_shape should be a tupe of lenght 2. Invalid tuple lenght ({len(resize_shape)} has been provided.)")
            elif not all(isinstance(x, numbers.Integral) for x in resize_shape):
                raise ValueError(f"All the values inside resize_shape parameter should be of type int. Invalid type has been provided")
            else:
                self.resize_shape = resize_shape
        else:
            self.resize_shape = None

        if not isinstance(flatten, bool):
            raise ValueError(f"flatten parameter should be of type bool. invalid type ({type(flatten)}) has been provided.")
        else:
            self.flatten = flatten

        self._set_model(model)


    def _extract_final_estimator(self, model) -> BaseEstimator:
        """
        Recursively unwraps common scikit-learn meta-estimators and pipelines
        to obtain the final estimator.

        Raises:
            ValueError: for unsupported wrappers 
        """
        from sklearn.pipeline import Pipeline
        from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
        from sklearn.ensemble import VotingClassifier, StackingClassifier,BaggingClassifier, AdaBoostClassifier
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.multioutput import MultiOutputClassifier, ClassifierChain
        from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
        from sklearn.feature_selection import RFE, RFECV
        if isinstance(model, (GridSearchCV, RandomizedSearchCV)):
            return self._extract_final_estimator(model.best_estimator_)

        elif isinstance(model, Pipeline):
            return self._extract_final_estimator(model.steps[-1][1])

        elif isinstance(model, CalibratedClassifierCV):
            return self._extract_final_estimator(model.base_estimator)

        elif isinstance(model, (OneVsRestClassifier, OneVsOneClassifier)):
            return self._extract_final_estimator(model.estimator)

        elif isinstance(model, (RFE, RFECV)):
            return self._extract_final_estimator(model.estimator_)

        elif isinstance(model, (VotingClassifier, StackingClassifier,
                                BaggingClassifier, AdaBoostClassifier)):
            return model

        elif isinstance(model, (MultiOutputClassifier, ClassifierChain)):
            raise ValueError("Multi-output classifiers are not supported.")

        return model


    def _set_model(self, model) -> None:
        from sklearn.utils.validation import check_is_fitted
        from sklearn.exceptions import NotFittedError

        estimator = self._extract_final_estimator(model)

        if not isinstance(estimator, BaseEstimator):
            raise TypeError("Model is not a valid scikit-learn estimator.")

        if not isinstance(estimator, ClassifierMixin):
            raise TypeError("Model is not a classifier.")

        try:
            check_is_fitted(estimator)
        except NotFittedError as e:
            raise ValueError("Model must be fitted before validation.") from e
        
        self.model = model
        
        if self._n_classes is None:
            if hasattr(estimator, "classes_"):
                self._n_classes = len(estimator.classes_)
            else:
                try:
                    dummy_input = np.zeros((1, estimator.n_features_in_))
                except AttributeError:
                    raise ValueError("Cannot determine input feature size. Model must expose `n_features_in_` or `classes_`.")
                try:
                    probs = estimator.predict_proba(dummy_input)
                    self._n_classes = probs.shape[1]
                except AttributeError:
                    try:
                        scores = estimator.decision_function(dummy_input)
                        self._n_classes = scores.shape[0] if scores.ndim == 1 else scores.shape[1]
                    except AttributeError:
                        raise ValueError("Cannot determine number of classes from model. Please provide them explicitely.")
            if self._n_classes == 2:
                self._n_classes = 1


    def _resize_image_tf(self, image: np.ndarray) -> np.ndarray:
        """
        Resize a grayscale or RGB image using TensorFlow.

        Parameters:
            image (np.ndarray): Input image (H, W) or (H, W, C)
            reshape_size (tuple): New shape (new_height, new_width)
            method (str): Resize method: 'bilinear', 'nearest', 'bicubic', 'lanczos3', etc.

        Returns:
            np.ndarray: Resized image with shape (new_height, new_width) or (new_height, new_width, C)
        """
        import tensorflow as tf
        img = image.copy()
        img = tf.convert_to_tensor(img, dtype=tf.float32)

        if img.ndim == 2:
            img = tf.expand_dims(img, axis=-1)

        img = tf.expand_dims(img, axis=0)

        if self.resize_method is not None:
            resized = tf.image.resize(img, self.resize_shape, method=self.resize_method)
        else: 
            resized = tf.image.resize(img, self.resize_shape)

        resized = tf.squeeze(resized, axis=0)

        if resized.shape[-1] == 1:
            resized = tf.squeeze(resized, axis=-1)

        return resized.numpy()


    def get_pred(self, image: np.ndarray) -> int:
        """
        Predict class label of a single input image (ndarray with ndim > 1).
        
        Tries raw input first, then falls back to flattened input.

        Parameters:
        - image (np.ndarray): Multi-dimensional image (ndim > 1).

        Returns:
        - int: Predicted class label.
        """
        if self.resize_shape is not None:
            image = self._resize_image_tf(image)

        if not self.flatten:
            input_data = image.reshape(1, *image.shape)
            try:
                pred = self.model.predict(input_data)
                return int(pred[0])
            except Exception:
                pass

        flat_input = image.copy()
        flat_input = image.flatten().reshape(1, -1)

        estimator = self._extract_final_estimator(self.model)
        n_features = getattr(estimator, "n_features_in_", None)
        if n_features is not None and flat_input.shape[1] != n_features:
            raise ValueError(f"After flattening, input features shape ({flat_input.shape[1]}) do not match expected shape({n_features}).")
        try:
            pred = self.model.predict(flat_input)
            return int(pred[0])
        except Exception as e:
            raise RuntimeError("Prediction failed with both raw and flattened inputs. "
                               "Check if input shape matches model expectations.") from e


    def get_score(self, image: ndarray, target: Optional[int] = None) -> Union[float, tuple[float, float]]:
        """
        Return the score(s) for the predicted and optionally the target class.

        Parameters:
        - image (ndarray): Multi-dimensional input image (ndim > 1)
        - target (Optional[int]): Optional target class index for score comparison

        Returns:
        - float: score for predicted class
        - tuple(float, float): (predicted class score, target class score) if target provided
        """
        if target is not None:
            if not isinstance(target, int):
                raise TypeError(f"target parameter is expected to be of type 'int'. The provided target type "
                                f"({type(target)}) is invalid.")
            else: 
                # Check target is positive
                if target < 0:
                    raise ValueError(f"'target' parameter should be a positive integer. "
                                     f"A negative integer ({target}) has been provided. Please provide a valid number.")
                # Check target in range of classes 
                if target not in range(self.get_n_classes()):
                    raise IndexError(f"The class specified is outiside the classes range. "
                                    f"Number of classes: {self.get_n_classes()}, target class specified: {target}")
        scores = None

        if self.resize_shape is not None:
            image = self._resize_image_tf(image)
        
        try:
            scores = self._get_scores(self.model, image)
        except Exception:
            estimator = self._extract_final_estimator(self.model)
            scores = self._get_scores(estimator, image)

        if scores is None:
            raise RuntimeError("Prediction failed with both raw and flattened inputs. "
                            "Check if input shape matches model expectations.")

        else:
            pred_class = scores.argmax(axis=1)[0]
            pred_score = float(scores[0, pred_class])

            if target is None:
                return pred_score
            else:
                target_score = float(scores[0, target])
                return pred_score, target_score


    def _get_scores(self, model, image: ndarray) -> ndarray:
        if not self.flatten:
            input_data = image.reshape(1, *image.shape)
            try:
                score = self._predict_scores(model, input_data)
                return score
            except Exception:
                pass

        flat_input = image.copy()
        flat_input = image.flatten().reshape(1, -1)

        estimator = self._extract_final_estimator(self.model)
        n_features = getattr(estimator, "n_features_in_", None)
        if n_features is not None and flat_input.shape[1] != n_features:
            raise ValueError(f"After flattening, input features shape ({flat_input.shape[1]}) do not match expected shape({n_features}).")
        try:
            score = self._predict_scores(model, flat_input)
            return score
        except Exception as e:
            raise RuntimeError("Prediction failed with both raw and flattened inputs. "
                               "Check if input shape matches model expectations.") from e 


    def _predict_scores(self, model, image) -> ndarray:
        """
        Helper function to get prediction scores as probabilities or decision function output.
        """
        if hasattr(model, "predict_proba"):
            return model.predict_proba(image)
        elif hasattr(model, "decision_function"):
            scores = model.decision_function(image)
            # decision_function can return shape (n_samples,) for binary classifiers, so expand dims
            if scores.ndim == 1:
                scores = scores[:, None]
            return scores
        else:
            raise RuntimeError("Estimator does not support probability or decision function outputs.")




class TorchModelWrapper(BaseModelWrapper):
    def __init__(self, model: nn.Module, 
                 transform: Optional[Callable[[Any], Tensor]] = None, 
                 **kwargs) -> None :
        super().__init__()
        self._output_is_logit = None
        self.device = self._get_device()

        if transform is not None: 
            self._set_transformation(transform)
        else:
            self.transform = None

        # Allow for parameters passed in the init method for advanced usage
        n_classes = kwargs.get("n_classes", None)
        logit_output = kwargs.get("logit_output", None)

        if n_classes is not None:
            if not isinstance(n_classes, numbers.Integral):
                raise ValueError(f"n_classes parameter should be of type int. Invalid type({type(n_classes)}) has been provided.")
            else:
                self._n_classes = n_classes

        if logit_output is not None:
            if not isinstance(logit_output, bool):
                raise ValueError(f"logit_output parameter should be of type bool. Invalid type({type(n_classes)}) has been provided.")
            else:
                self._output_is_logit = logit_output

        self._set_model(model)


    def _set_transformation(self, transform):
        '''Check and set the transformation.'''
        if not callable(transform):
            raise TypeError(f"Expected a callable transformation, got {type(transform)} instead.")
        self.transform = transform


    def _get_device(self):
        '''Return the device for torch models.'''
        import torch
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")


    def _set_model(self, model) -> None :
        '''Check and set the model. Also set _n_classes.'''
        if not isinstance(model, nn.Module):
            raise TypeError(f"The provided model is not a Torch model (nn.Module). " 
                            f"The provided type is {type(model)}. Please pass a Torch model or use a different wrapper.")
         
        self.model = model.to(self.device)
        self.model.eval()
        # Set number of classes
        if self._n_classes is None or self._output_is_logit is None:
            found_linear = False
            for module in reversed(list(self.model.modules())):
                if isinstance(module, nn.Linear):
                    self._n_classes = module.out_features
                    found_linear = True
                    break
                if isinstance(module, nn.Sigmoid):
                    self._output_is_logit = False
                elif isinstance(module, (nn.ReLU, nn.Dropout, nn.BatchNorm1d, nn.Flatten)):
                    continue
                else:
                    # Unknown type after output layer
                    if not found_linear:
                        raise TypeError(f"Unexpected module in final layers: {type(module).__name__}. "
                                        f"Expected nn.Linear or nn.Sigmoid near the end.")
            if not found_linear:
                raise TypeError("No nn.Linear layer found in model's final layers. The model might not be a classifier.")


    def get_pred(self, image: Tensor) -> int:
        if not isinstance(image, Tensor):
            raise TypeError(f"image is expected to be of type 'torch.Tensor'. The provided image "
                            f"type ({type(image)}) is invalid.")
        import torch
        from torchvision.transforms.functional import to_pil_image  
        if self.transform is not None:
            pil_image = to_pil_image(image)
            image = self.transform(pil_image)
        image = image.unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(image)
            if self.is_multiclass():
                # Multiclass classification
                probs = torch.softmax(output, dim=1)
                pred = torch.argmax(probs).item()
                return int(pred)
            else:
                # Binary classification
                if self._output_is_logit:
                    probs = torch.sigmoid(output)
                else:
                    probs = output

                pred = int(probs[0].item() > 0.5)
                return pred


    def get_score(self, image: Tensor, target: Optional[int] = None) -> Union[float, tuple[float, float]] :
        if not isinstance(image, Tensor):
            raise TypeError(f"image parameter is expected to be of type 'torch.Tensor'. The provided image "
                            f"type ({type(image)}) is invalid.")
        if target is not None:
            if not isinstance(target, numbers.Integral):
                raise TypeError(f"target parameter is expected to be of type 'int'. The provided target type "
                                f"({type(target)}) is invalid.")
            else: 
                # Check target is positive
                if target < 0:
                    raise ValueError(f"'target' parameter should be a positive integer. "
                                     f"A negative integer ({target}) has been provided. Please provide a valid number.")
                # Check target in range of classes 
                if target not in range(self.get_n_classes()):
                    raise IndexError(f"The class specified is outiside the classes range. "
                                    f"Number of classes: {self.get_n_classes()}, target class specified: {target}")
        import torch
        from torchvision.transforms.functional import to_pil_image  
        if self.transform is not None:
            pil_image = to_pil_image(image)
            image = self.transform(pil_image)
        image = image.unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(image)
            if self.is_multiclass():
                # Multiclass classification
                probs = torch.softmax(output, dim=1)
                pred = torch.argmax(probs).item()
                pred = int(pred)
                pred_score = probs[0][pred].item()

                if target is not None:
                    target_score = probs[0][target].item()
                    return pred_score, target_score
                else:
                    return pred_score
            else:
                # Binary classification 
                if self._output_is_logit:
                    probs = torch.sigmoid(output)
                else:
                    probs = output
                    
                pred_score = probs[0].item()
                return pred_score
