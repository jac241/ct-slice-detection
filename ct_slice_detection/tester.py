from pathlib import Path

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

from ct_slice_detection.inout.dataloader import DataLoader
from ct_slice_detection.inout.parameters import parse_inputs
from ct_slice_detection.models import Models
from ct_slice_detection.utils.generic_utils import Fold
from ct_slice_detection.utils.testing_utils import *


def main():

    args = parse_inputs()

    print(args)

    # GPU allocation options
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = args.cuda_devices
    set_session(tf.Session(config=config))



    baseModel = Models(args.model_name)

    if args.do_crossval:
        test_data = DataLoader(args)
        test_data.load_train_val_split(do_cross_val=True)
        cross_val_data = test_data.cross_val_data.item()
        print(cross_val_data.keys())
        num_splits = args.n_splits
        for i in range(num_splits):
            train_index = cross_val_data[i]['train']
            val_index = cross_val_data[i]['val']
            test_data.split_data(train_index, val_index)

            evaluate_fold(args, baseModel, i, num_splits, test_data)
    elif args.do_predefined_crossval:
        test_data = DataLoader(args)
        test_data.load_data()
        for fold_index in range(args.n_splits):
            fold = Fold(root_path=Path(args.predefined_folds_dir),
                        fold_index=fold_index,
                        subject_ids=list(test_data.names_val))
            train_indices = fold.get_train_indices()
            test_indices = fold.get_test_indices()
            test_data.split_data(train_indices, test_indices)
            evaluate_fold(args, baseModel, fold_index, args.n_splits, test_data)

    else:
        test_data = DataLoader(args)

        if args.test_mode == 'eval':
            test_data.load_train_val_split()
            test_data.split_data(test_data.train_idx, test_data.val_idx)
        else:
            test_data.load_data()

        # Setup model
        modelwrapper = baseModel(name=args.model_name,
                                   config=args,
                                   input_shape=args.model_input_shape,
                                   data_loader=test_data
                                   )

        modelwrapper.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        modelwrapper.load()

        predict_and_evaluate(args, test_data, modelwrapper)


def evaluate_fold(args, baseModel, i, num_splits, test_data):
    model_name = args.model_name + '_cv_' + str(i + 1) + '_of_' + str(
        num_splits)
    modelwrapper = baseModel(name=model_name,
                             config=args,
                             input_shape=args.model_input_shape,
                             data_loader=test_data
                             )
    modelwrapper.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    modelwrapper.load()
    with open(os.path.join(args.model_path, "time_log.txt"), "a") as file:
        start_time = time.time()
        predict_and_evaluate(args, test_data, modelwrapper,
                             suffix='_cv_' + str(i))
        end_time = time.time()
        dt = end_time - start_time
        n = len(test_data.x_val)
        file.write(
            "{}, total time for {}: {} , (avg: {}) \n ".format(model_name, n,
                                                               dt, dt / n))


if __name__ == '__main__':
    main()
