import logging

from models.deep_sound import DeepSound
from experiments.settings import random_seed
from data.make_dataset import main
from experiments.base import Experiment
from features.feature_factories import FeatureFactory_RawAudioData

from yaer.base import experiment


logger = logging.getLogger('yaer')


def get_model_instance(variable_params):
    return DeepSound(input_size=1800,
                     output_size=6,
                     n_epochs=1500,
                     batch_size=10,
                     training_reshape=True,
                     set_sample_weights=True,
                     feature_scaling=True)


@experiment()
def deep_sound():
    """ Experiment with Deep Sound architecture.
    """
    window_width = 0.3
    window_overlap = 0.5
    X, y = main(window_width=window_width,
                window_overlap=window_overlap,
                include_movement_magnitudes=True,
                audio_sampling_frequency=6000)

    e = Experiment(get_model_instance,
                   FeatureFactory_RawAudioData,
                   X, y,
                   window_width=window_width,
                   window_overlap=window_overlap,
                   name='deep_sound',
                   manage_sequences=True,
                   use_raw_data=True)

    e.run()
