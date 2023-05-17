## Justin Wong
## Unit tests for FIMUS implementation

import fimus
import random

D = [
    [27,  'MS',  85, 'L'],
    [45,  None, 145, 'P'],
    [42, 'PhD', 145, 'P'],
    [25,  'MS',  85, 'L'],
    [50, 'PhD', 146, 'P'],
    [28,  'MS',  85, 'L'],
    [38, 'PhD', 140, 'P'],
    [43, 'PhD', 148, None],
    [44, 'PhD', 146, 'P'],
    [None, 'MS', 86, 'L'],
    [42, 'PhD', 142, 'P'],
    [26,  'MS',  84, 'L'],
    [42, 'PhD', None, 'P'],
    [25,  'MS',  86, 'L'],
    [43, 'PhD', 143, 'P'],    
]

B = [
    [0, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 0, 0],
    [1, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
]

DG = [
    [(25, 29),  'MS', (84, 91),   'L'],
    [(45, 49),  None, (140, 147), 'P'],
    [(40, 44), 'PhD', (140, 147), 'P'],
    [(25, 29),  'MS', (84, 91),   'L'],
    [(50, 54), 'PhD', (140, 147), 'P'],
    [(25, 29),  'MS', (84, 91),   'L'],
    [(35, 39), 'PhD', (140, 147), 'P'],
    [(40, 44), 'PhD', (148, 155), None],
    [(40, 44), 'PhD', (140, 147), 'P'],
    [None,      'MS', (84, 91),   'L'],
    [(40, 44), 'PhD', (140, 147), 'P'],
    [(25, 29),  'MS', (84, 91),   'L'],
    [(40, 44), 'PhD', None,       'P'],
    [(25, 29),  'MS', (84, 91),   'L'],
    [(40, 44), 'PhD', (140, 147), 'P'],
]

C = [
    [0, 0, 0, 0, 0, 5, 0, 5, 0, 0, 5, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 6, 0, 4, 1, 0, 5],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1],
    [5, 0, 0, 0, 0, 0, 0, 6, 0, 0, 6, 0],
    [0, 1, 6, 0, 1, 0, 0, 0, 6, 1, 0, 7],
    [5, 0, 0, 0, 0, 6, 0, 0, 0, 0, 6, 0],
    [0, 1, 4, 1, 1, 0, 6, 0, 0, 0, 0, 7],
    [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [5, 0, 0, 0, 0, 6, 0, 6, 0, 0, 0, 0],
    [0, 1, 5, 1, 1, 0, 7, 0, 7, 0, 0, 0],
]

S = [
    [
        [1.000, 0.000, 0.000, 0.000, 0.000],
        [0.000, 0.264, 0.215, 0.258, 0.264],
        [0.000, 0.240, 0.294, 0.225, 0.240],
        [0.000, 0.263, 0.206, 0.269, 0.263],
        [0.000, 0.264, 0.215, 0.258, 0.264]
    ],
    [
        [1, 0],
        [0, 1]
    ],
    [
        [1.000, 0.000, 0.000],
        [0.000, 1.000, 0.745],
        [0.000, 0.745, 1.000]
    ],
    [
        [1, 0],
        [0, 1]
    ]
]

K = [
    [0.000, 0.707, 0.721, 0.707],
    [0.707, 0.000, 0.707, 0.707],
    [0.721, 0.707, 0.000, 0.707],
    [0.707, 0.707, 0.707, 0.000]
]

### functionality unit tests

def test_missing():
    empty = [[0 for col in D[0]]
             for row in D]
    assert fimus.missing(D) == B
    assert not fimus.missing(D) == empty
    
def test_cols():
    cols = [
        [27, 45, 42, 25, 50, 28, 38, 43, 44, None, 42, 26, 42, 25, 43],
        ['MS', None, 'PhD', 'MS', 'PhD', 'MS', 'PhD', 'PhD', 'PhD', 'MS', 'PhD', 'MS', 'PhD', 'MS', 'PhD'],
        [85, 145, 145, 85, 146, 85, 140, 148, 146, 86, 142, 84, None, 86, 143],
        ['L', 'P', 'P', 'L', 'P', 'L', 'P', None, 'P', 'L', 'P', 'L', 'P', 'L', 'P'],
    ]
    assert list(fimus.cols(D)) == cols
    assert not list(fimus.cols(D)) == D
    
def test_zeros():
    arr = [
        [0, 0, 0],
        [0, 0, 0]
    ]
    assert fimus.zeros(2, 3) == arr
    
def test_numeric():
    assert fimus.is_numeric(7)
    assert fimus.is_numeric(-3.5)
    assert fimus.is_numeric([1, 5, 4])
    assert fimus.is_numeric([i for i in range(3)])
    assert fimus.is_numeric([None, 4, 5, 6, 7, 8, 9])
    assert not fimus.is_numeric(['a', 'b', 'c'])
    
def test_filtered():
    assert fimus.filtered(D[1]) == [45, 145, 'P']
    
def test_bin_size():
    assert fimus.bin_size(list(fimus.cols(D))[2]) == 8
    
def test_generalise():
    assert fimus.generalise(D) == DG
    
def test_domain():
    cols = list(fimus.cols(DG))
    a1 = cols[0]
    a2 = cols[1]
    domain_a1 = [(25,29), (35,39), (40,44), (45,49), (50,54)]
    domain_a2 = ['MS', 'PhD']
    assert fimus.domain(a1) == domain_a1
    assert not fimus.domain(a1) == domain_a1 + [None]
    assert fimus.domain(a2) == domain_a2
    
def test_coappearances():
    assert fimus.coappearances(DG) == C
    
def test_normalize():
    S_age = [
        [1.000, 0.000, 0.000, 0.000, 0.000],
        [0.000, 1.000, 0.816, 0.979, 1.000],
        [0.000, 0.816, 1.000, 0.766, 0.816],
        [0.000, 0.979, 0.766, 1.000, 0.979],
        [0.000, 1.000, 0.816, 0.979, 1.000]
    ]
    s_norm = fimus.normalize(S_age)
    s_norm_rounded = rounded(s_norm)   
    assert s_norm_rounded == S[0]
    
    error_thresh = 0.01     # small allowance for floats nearly adding to 1
    for row in s_norm:
        assert abs(sum(row) - 1) < error_thresh
    
def test_expected():
    obs = [
        [19, 32, 83, 97, 48],
        [ 2,  6, 16, 42, 26],
        [ 0,  1,  3, 21, 10],
    ]
    exp = [
        [14.4, 26.8, 70.1, 110.0, 57.7],
        [ 4.8,  8.8, 23.1,  36.3, 19.0],
        [ 1.8,  3.4,  8.8,  13.8,  7.2]
    ]
    assert rounded(fimus.expected(obs), 1) == exp
    
def test_records_in_range():
    DN = fimus.records_in_range((25,29), 0, D)
    exp_DN = [D[row] for row in (0, 3, 5, 11, 13)]
    assert DN == exp_DN 

def test_similarity_execution():
    '''Asserts similarity matrix per column of data is of size domain x domain.'''
    for col in fimus.cols(DG):
        _domain = fimus.domain(col)
        s_test = fimus.similarity(DG,
                                  _domain,
                                  C)
        assert (len(s_test) == len(_domain) and
                len(s_test[0]) == len(_domain))
    
def test_correlation_execution():
    '''Asserts correlation matrix for data is of size numcols x numcols.'''
    K_test = fimus.correlation(DG, C)
    assert (len(K_test) == len(DG[0]) and 
            len(K_test[0]) == len(DG[0]))
    
def test_CSR_execution():
    similarity_threshold = 0.2
    i = 9
    j = 0
    _domain = fimus.domain(list(fimus.cols(DG))[j])
    val_to_impute = fimus.CSR(DG[i], 
                              j,
                              C, 
                              S, 
                              K, 
                              similarity_threshold,
                              DG)
    assert val_to_impute in _domain
    
def test_deeper_CSR_execution():
    _min, _max = (140, 147) #(25, 29)
    i, j = (12, 2) #(9, 0)

    DN = fimus.records_in_range((_min, _max), j, D)
    C_new = fimus.coappearances(DN)
    S_new = [fimus.similarity(DN, d, C_new)
             for d in (fimus.domain(col) 
                       for col in fimus.cols(DN))]
    K_new = fimus.correlation(DN, C_new)
    similarity_threshold = 0.2
    
    x = fimus.CSR(D[i],
                  j,
                  C_new,
                  S_new,
                  K_new,
                  similarity_threshold,
                  DN)
    assert (type(x) == int and
            x >= _min and
            x <= _max)

def test_steps_2_thru_4():
    D_new = fimus.steps_2_thru_4(D, 0.2)
    D_exp = [row.copy() for row in D]
    D_exp[1][1] = 'PhD'
    assert D_exp == D_new

def test_main_execution():
    D_final = fimus.main(D)
    for col in fimus.cols(D_final):
        col_type = type(col[0])
        assert not col_type == None
        for val in col:
            assert type(val) == col_type
            
### accuracy tests, where algorithm results slightly differ from expected

def test_similarity_accuracy():
    
    S_test = [rounded(
                fimus.normalize(
                    fimus.similarity(DG,
                                     fimus.domain(col),
                                     C)))
              for col in fimus.cols(DG)]
    
    for s, s_test in zip(S, S_test):
        if not s == s_test:
            print(test_similarity_accuracy, 'failed by', error(s, s_test))
        
def test_correlation_accuracy():
    K_test = fimus.correlation(DG, C)
#    assert K = K_test
    if not K == K_test:
        print(test_correlation_accuracy, 'failed by', error(K, K_test))
    
def test_CSR_accuracy():
    # ideally looks at votes for runner-up accuracy, but that requires restructuring fimus.CSR
    similarity_threshold = 0.2
    locs = ((i,j) for i in range(len(B))
            for j in range(len(B[0]))
            if B[i][j] == True)
    
    expected = ['PhD', 'P', (25,29), (140,147)] 
    observed = [fimus.CSR(DG[i], 
                          j,
                          C, 
                          S, 
                          K, 
                          similarity_threshold,
                          DG)
                for i, j in locs]
       
    successes = [True if obs == exp else False
                 for obs, exp in zip(observed, expected)]
    print(test_CSR_accuracy, 'accuracy:', 
          percent(compare_categoric_results(list(zip(observed, expected)))))

def test_deeper_CSR_accuracy():
    _range = (140, 147) #(25, 29)
    i, j = (12, 2) #(9, 0)

    DN = fimus.records_in_range(_range, j, D)
    C_new = fimus.coappearances(DN)
    S_new = [fimus.similarity(DN, d, C_new)
             for d in (fimus.domain(col) 
                       for col in fimus.cols(DN))]
    K_new = fimus.correlation(DN, C_new)
    similarity_threshold = 0.2
    
    x = fimus.CSR(D[i],
                  j,
                  C_new,
                  S_new,
                  K_new,
                  similarity_threshold,
                  DN)
    print(test_deeper_CSR_accuracy, 'exp: 142 ,', 'obs:', x)
    
def test_on_sample_data():
    locs = [(i,j) for i in range(len(B))
            for j in range(len(B[0]))
            if B[i][j] == True]
    
    expected = ['PhD', 'P', 25, 142]
    
    print('Results on sample data from paper:')
    test_on_data(D, expected, locs)
     
def test_on_yeast_data():
    '''Test fimus.main against established dataset.
    https://archive.ics.uci.edu/ml/datasets/Yeast'''
    data = []
    with open('yeast.data', 'r') as file:
        line = file.readline()
        while line:
            row = []
            for val in line.split()[1:]:
                try:
                    val = round( float(val) * 100)
                except ValueError:
                    pass
                row.append(val)
            data.append(row)
            line = file.readline()

    num_records = 150   # could implement to accept 0.5 to reperesent 50% of cases
    D = random.choices(data, k=num_records)
    
    num_deletions = 5
    locs = []
    while len(locs) < num_deletions:
        i = random.randint(0, len(D)-1)
        j = random.randint(0, len(D[0])-1)
        loc = (i, j)
        if not loc in locs:
            locs.append(loc)
    
    expected = []
    for i, j in locs:
        expected.append(D[i][j])
        D[i][j] = None
    
    print('Results on yeast dataset:')
    test_on_data(D, expected, locs)
    
def test_demo_data():
    D = [
        ['red',  3, 10],
        ['blue', 6, 1],
        ['blue', 7, 2],
        ['red',  1, 8],
        ['?',  '?', 9],
        ['blue', '?', '?'],
    ]
    for row in D:
        print(row)
        
        
    for i in range(len(D)):
        for j in range(len(D[0])):
            if D[i][j] == '?':
                D[i][j] = None
    
    D_final = fimus.main(D)
    
    print()
    for row in D_final:
        print(row)
    
def unit_tests():
    test_missing()
    test_cols()
    test_zeros()
    test_numeric()
    test_filtered()
    test_bin_size()
    test_generalise()
    test_domain()
    test_coappearances()
    test_normalize()
    test_expected()
    test_records_in_range()
    test_similarity_execution()
    test_correlation_execution()
    test_CSR_execution()
    test_deeper_CSR_execution()
    test_steps_2_thru_4()
    test_main_execution()
    
def accuracy_tests():
    test_similarity_accuracy()
    test_correlation_accuracy()
    test_CSR_accuracy()
    test_deeper_CSR_accuracy()
    test_on_sample_data()
    test_on_yeast_data()
    
def all_tests():
    unit_tests()
    print('Unit tests passed.')
    accuracy_tests()
    
def test_on_data(D, expected, locs):
    
    D_imputed = fimus.main(D)
    observed = [D_imputed[i][j] for i, j in locs]
    
    numeric_results = []
    categoric_results = []
    for obs, exp in zip(observed, expected):
        if fimus.is_numeric(exp):
            numeric_results.append((obs, exp))
        else:
            categoric_results.append((obs, exp))
    
    i_of_agreement = d2(numeric_results)
    rmse = RMSE(numeric_results)
    categoric_accuracy = compare_categoric_results(categoric_results)
    
    print('expected', expected)
    print('observed', observed)
    print('numeric accuracy:', percent(i_of_agreement))
    print('\t error:', rmse)
    print('categorical accuracy:', percent(categoric_accuracy))
    print()
    
def rounded(arr, precision=3):
    '''Returns copy of 2D arr with elements rounded to given precision.'''
    return [[_round(val, precision) for val in row]
            for row in arr] 

def _round(val, precision=3):
    '''Rounds num to given decimal places.'''
    return round(val * 10**precision) / 10 ** precision
    
def error(a, b):
    '''Estimate for difference between matrices a, b of same size.
    Sum of sqaured diffs.'''
    err = 0
    for r in range(len(a)):
        for c in range(len(a[0])):
            err += (a[r][c] - b[r][c]) ** 2
    return err ** 0.5

def RMSE(numeric_results):
    '''Returns root mean squared error.
    Used to assess imputation accuracy.
    0 means best accuracy, inf worst.'''
    if len(numeric_results) == 0:
        return None
    
    diffs = [(obs - exp)**2 for obs, exp in numeric_results]
    return ( ( 1 / len(diffs) ) * sum(diffs) ) ** 0.5
    
def d2(numeric_results):
    '''Returns index of agreement.
    Used to assess imputation accuracy.
    0 means low resemblance, 1 means better resemblance.'''
    if len(numeric_results) == 0:
        return None
    avg_obs = avg([obs for obs, exp in numeric_results])
    avg_exp = avg([exp for obs, exp in numeric_results])
    
    num = sum((
        (obs - exp) ** 2 
        for obs, exp in numeric_results))
    potential_error = sum((
        ( abs(obs - avg_obs) + abs(exp - avg_exp) ) ** 2
        for obs, exp in numeric_results))
    
    return 1 - (num / potential_error)

def compare_categoric_results(categoric_results):
    if len(categoric_results) == 0:
        return None

    successes = [True if obs == exp else False
                 for obs, exp in categoric_results]
    return sum(successes) / len(successes)

def avg(lis):
    return sum(lis) / len(lis)

def percent(f):
    '''Returns str of decimal as percent rounded to nearest tenth.'''
    return None if f is None else str( _round(f * 100, 1) ) + ' %'

if __name__ == '__main__':
    all_tests()