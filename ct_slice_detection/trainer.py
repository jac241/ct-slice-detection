import os
from pathlib import Path

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from sklearn.model_selection import KFold

from ct_slice_detection.inout.dataloader import DataLoader
from ct_slice_detection.inout.parameters import parse_inputs
from ct_slice_detection.models import Models
from ct_slice_detection.utils.generic_utils import Fold
from ct_slice_detection.utils.training_utils import PreviewOutput


def cross_validate(baseModel, args):
    pretrained_model = None

    trainer_data = DataLoader(args)
    kf = KFold(n_splits=args.n_splits, random_state=args.random_state, shuffle=True)
    num_samples = trainer_data.get_num_samples()
    for idx, (train_index, val_index) in enumerate(kf.split(list(range(trainer_data.num_samples)))):
        print('cross validation step {} of {}'.format(idx + 1, args.n_splits))
        print(val_index)

        train_fold(args, baseModel, idx, train_index, trainer_data, val_index)


def train_fold(args, baseModel, fold_index, train_index, trainer_data, val_index):
    trainer_data.split_data(train_index, val_index)
    trainer_data.update_crossval_data(fold_index)
    trainer_data.save_train_val_split(True)
    if args.preview_generator_output:
        trainer_data.preview_generator_output()
    # Setup model
    model_name = args.model_name + '_cv_' + str(fold_index + 1) + '_of_' + str(
        args.n_splits)
    modelwrapper = baseModel(name=model_name,
                             config=args,
                             input_shape=args.model_input_shape,
                             data_loader=trainer_data
                             )
    if args.preview_training_output:
        modelwrapper.callbacks.append(PreviewOutput(trainer_data, 10, args))
    print(modelwrapper.model.summary())
    try:
        modelwrapper.train_generator()

    except KeyboardInterrupt:
        pass
    modelwrapper.save()


def cross_validate_with_predefined_groups(baseModel, args):
    trainer_data = DataLoader(args)
    name_loader = DataLoader(args)
    name_loader.load_data()

    for index in range(4):
        # seg h5 file indicies --> subject ids
        fold = Fold(
            root_path=Path(args.predefined_folds_dir),
            fold_index=index,
            subject_ids=list(name_loader.names_val),
            h5_datafile_path=args.h5_file_with_subject_ids_for_indices,
            npz_mips_file_path=args.dataset_path,
        )

        train_fold(
            args, baseModel,
            fold_index=index,
            train_index=fold.get_train_indices(),
            trainer_data=trainer_data,
            val_index=fold.get_val_indices()
        )



def main():

    args = parse_inputs()

    print(args)
    # GPU allocation options
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # config.gpu_options.visible_device_list = args.cuda_devices
    set_session(tf.Session(config=config))

    #Handle restarting and resuming training
    if args.restart:
        print('Restarting training from scratch.')
        os.system('rm -rf {}'.format(args.model_path))

    if not os.path.isdir(args.model_path):
        os.system('mkdir -p {}'.format(args.model_path))
    else:
        print('Resuming training on model_path {}'.format(args.model_path))


    baseModel = Models(args.model_name)

    if args.do_crossval and args.do_predefined_crossval:
        raise Exception("Conflicting config parameters, both do_crossval and "
                        "do_predefined_crossval set, choose one")
    elif args.do_crossval:
        cross_validate(baseModel, args)
    elif args.do_predefined_crossval:
        cross_validate_with_predefined_groups(baseModel, args)
    else:
        trainer_data = DataLoader(args)
        trainer_data.split_data()

        if args.preview_generator_output:
            trainer_data.preview_generator_output()

        # Setup model
        modelwrapper = baseModel(name=args.model_name,
                                   config=args,
                                   input_shape=args.model_input_shape,
                                   data_loader=trainer_data
                                   )

        if args.preview_training_output:
            modelwrapper.callbacks.append(PreviewOutput(trainer_data,2, args))

        print(modelwrapper.model.summary())

        try:
            modelwrapper.train_generator()

        except KeyboardInterrupt:
            pass

        modelwrapper.save()





if __name__ == '__main__':
    main()
