# import logging

# from models import ablation_models as am
# from data.make_dataset import main
# from experiments.base import Experiment, PredictionExperiment
# from features import feature_factories as ff

# from yaer.base import experiment


# logger = logging.getLogger('yaer')


# def get_model_A_instance(variable_params):
#     return am.DeepFusionAblationA(
#         input_size_audio=variable_params['input_size_audio'],
#         output_size=6,
#         n_epochs=1500,
#         batch_size=10,
#         training_reshape=True,
#         set_sample_weights=True,
#         feature_scaling=True)




# @experiment()
# def ablation_model_A_validation():
#     """ Experiment with sound head on validation data.
#     """
#     window_width = 0.3
#     window_overlap = 0.5
#     X, y = main(window_width=window_width,
#                 window_overlap=window_overlap,
#                 include_movement_magnitudes=True,
#                 audio_sampling_frequency=6000)

#     e = Experiment(get_model_A_instance,
#                    ff.FeatureFactory_AllRawData,
#                    X, y,
#                    window_width=window_width,
#                    window_overlap=window_overlap,
#                    name='ablation_model_A_validation',
#                    manage_sequences=True,
#                    use_raw_data=True,
#                    model_parameters_grid={'input_size_audio': [(None, 1800, 1)]})

#     e.run()








# @experiment()
# def ablation_model_A_test():
#     """ Experiment with sound head on test data.
#     """
#     window_width = 0.3
#     window_overlap = 0.5
#     X, y = main(window_width=window_width,
#                 window_overlap=window_overlap,
#                 include_movement_magnitudes=True,
#                 audio_sampling_frequency=6000)

#     e = PredictionExperiment(
#         get_model_A_instance,
#         ff.FeatureFactory_AllRawData,
#         X, y,
#         window_width=window_width,
#         window_overlap=window_overlap,
#         name='ablation_model_A_test',
#         manage_sequences=True,
#         use_raw_data=True,
#         model_parameters_grid={'input_size_audio': [(None, 1800, 1)]}
#     )

#     e.run()




