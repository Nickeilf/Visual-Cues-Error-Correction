# First the basic types
from .npy import NumpyDataset
from .kaldi import KaldiDataset
from .imagefolder import ImageFolderDataset
from .text import TextDataset
from .numpy_sequence import NumpySequenceDataset
from .label import LabelDataset
from .shelve import ShelveDataset
from .objdet import ObjectDetectionsDataset


# Second the selector function
def get_dataset(type_):
    return {
        'numpy': NumpyDataset,
        'numpysequence': NumpySequenceDataset,
        'kaldi': KaldiDataset,
        'imagefolder': ImageFolderDataset,
        'text': TextDataset,
        'label': LabelDataset,
        'shelve': ShelveDataset,
        'objectdetections': ObjectDetectionsDataset,
    }[type_.lower()]


# Should always be at the end
from .multimodal import MultimodalDataset
