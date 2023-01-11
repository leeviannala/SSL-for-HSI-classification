from main import main
from argparse import Namespace


if __name__ == '__main__':
    dct = {
        'DHCN_LAYERS': 1, 
        'SAMPLE_PERCENTAGE': 5, 
        'DATASET': "IndianPines", 
        'CONV_SIZE': 3, 
        'ROT': True, 
        'MIRROR': True, 
        'H_MIRROR': 'full', 
        'GPU': '0', 
        'ROT_N': 1, 
        'MIRROR_N': 1, 
        'USE_HARD_LABELS_N': 1,
    }
    #datasets = ['KSC', 'PaviaU', 'IndianPines', 'Botswana']
    # datasets = ['IndianPines', 'Botswana']
    datasets = ['Botswana']
    # sample_percentages = [0.01, 0.05, 0.1, 1, 5, 10]
    sample_percentages = [1, 5, 10]
    use_hard_labels = [1, 0] 
    illegal_pairings = {'KSC':[], 'PaviaU':[], 'IndianPines':[0.01], 'Botswana':[0.01]}
    count = 0
    for ds in datasets:
        for sp in sample_percentages:
            for hl in use_hard_labels:
                if sp in illegal_pairings[ds]:
                    dot = False
                    continue
                else:
                    dot = True
                    count += 1
                if dot:
                    dct['DATASET'] = ds
                    dct['USE_HARD_LABELS_N'] = hl
                    dct['SAMPLE_PERCENTAGE'] = sp    
                    params = Namespace(**dct)
                    params.ROT = True if params.ROT_N == 1 else False
                    params.MIRROR = True if params.MIRROR_N == 1 else False
                    params.USE_HARD_LABELS = True if params.USE_HARD_LABELS_N == 1 else False
                    main(params)
    #print(count * 2880 / 60 / 60)
    #print(60*60)
                