from sklearn.metrics import accuracy_score


def eval_model(model, train_data, valid_data):
    """"""
    # train the model
    model.fit(*train_data)
    
    # test the model
    p = model.predict(valid_data[0])
    return accuracy_score(valid_data[1], p)


def instantiate_model(model_class, model_size):
    """"""
    model_size = int(model_size)
    if model_class == 'LogisticRegression':
        model = LogisticRegression()
    elif model_class == 'RandomForestClassifier':
        model = RandomForestClassifier(model_size, n_jobs=-1)
    elif model_class == 'MLPClassifier':
        model = MLPClassifier(
            (model_size,), learning_rate_init=0.001, early_stopping=True,
            learning_rate='adaptive'
        )
    else:
        raise ValueError(
            '[ERROR] only LogisticRegression, RandomForestClassifier, ' +
            'MLPClassifier are supported!'
        )
    return model


def setup_argparse():
    # setup argparser
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('classification_fn', type=str,
                        help='filename of the classification label data')
    parser.add_argument('feature_fn', type=str,
                        help='filename to the lyrics feature file')
    parser.add_argument('--out-fn', type=str, default='test.csv',
                        help='filename for the test output')
    parser.add_argument('--n-rep', type=int, default=5,
                        help='number of testing users')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = setup_argparse()
    
    # load run specific packages
    from itertools import chain
    from skopt import gp_minimize
    from skopt.space import Real, Integer
    from skopt.utils import use_named_args
     
    from ..utils import (load_csr_data,
                         prepare_feature,
                         get_all_comb)
    from .autotagging import (split_data,
                              get_model_instance,
                              full_factorial_design)
    
    # get full factorial design
    configs = full_factorial_design()
    
    # load the relevant data
    X, y = load_data(args.classification_fn, args.feature_fn)
    
    # run
    result = []
    for i, conf in enumerate(configs):
        print(conf)
        print('Running {:d}th / {:d} run...'.format(i+1, len(configs)))
        
        # prepare feature according to the design
        x = np.concatenate([v for k, v in X.items() if conf[k]], axis=1)
        for j in range(args.n_rep):
            # split data
            (Xtr, Xvl, Xts), (ytr, yvl, yts) = split_data(x, y)
            
            # find best model 
            model = get_model_instance(conf['model'], (Xtr, ytr), (Xvl, yvl))
            
            # prep test
            X_ = np.concatenate([Xtr, Xvl], axis=0)
            y_ = np.concatenate([ytr, yvl], axis=0)
            test = eval_model(model, (X_, y_), (Xts, yts))
            
            # register results
            metrics = {'trial': j, 'score': np.mean(test)}
            conf_copy = conf.copy()
            conf_copy.update(metrics)
            result.append(conf_copy)
            
    # save
    pd.DataFrame(result).to_csv(args.out_fn) 