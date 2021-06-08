## Justin Wong
## implementing FIMUS algorithm
## Paper at:
## https://www.researchgate.net/publication/259185364_FIMUS_A_Framework_for_Imputing_Missing_Values_Using_Co-appearance_Correlation_and_Similarity_Analysis
## Section 3 especially

import itertools

def pseudocode_FIMUS():
    '''Rough translation of algorithm in FIMUS paper.
    Compromise between paper's notation, python language, this implementation, and readability.
    Not to be executed; just pseudocode.'''
    D = [
        ['a', 7], #R1
        ['b', 4], #R2
        [None, 3], #R3
    ]
#    T = 0   # counter
#    N = len(D)

    # step-1
    B = missing(D)
    similarity_threshold = 0.2 # defaulted; user input from 0-1

    # step-2
    # turn numbers into ranges so everything is a category
    DG = generalise(D)
    
    domains = [domain(col) for col in cols(DG)]
    
    # step-3
    C = coappearances(DG)
    
    S = []
    for d in domains:
        # https://crpit.scem.westernsydney.edu.au/confpapers/CRPITV134Giggins.pdf
        S_d = similarity(DG, d, C) 
        S.append(normalize(S_d))

    K = correlation(DG, C)
    
    # step-4
    for i in range(len(DG)):
        for j in range(len(DG[0])):
            if B[i][j] == 1:
                # impute this missing value
                
                d = domains[j]
                # best vote
                x = CSR(DG[i],
                        d,
                        C,
                        S,
                        K,
                        similarity_threshold,
                        DG)

                if is_numeric(list(cols(D))[j]):
                    # another round with specific vals instead of generalised ranges
                    DN = records_in_range(x, # this category attr is a range
                                          j, # check this col
                                          D) # using original dataset
                    C_new = coappearances(DN)
                    S_new = [similarity(DN, d, C_new) 
                             for d in (domain(col) for col in cols(DN))]
                    K_new = correlation(DN, C_new)
                    domain_new = domain(list(cols(DN))[j])

                    # new x, value within range of prev x attr
                    x = CSR(D[i],
                            domain_new,
                            C_new,
                            S_new,
                            K_new,
                            similarity_threshold,
                            DN)
                # fill missing val in original using this best vote
                D[i][j] = x
                return  # we only want to fill in one missing value at a time
                        # bc authors said iterations like this increase accuracy
    # step-5
    T += 1
    
    DT_last = steps_2_thru_4(D)
    DT = steps_2_thru_4(DT_last)
    while not DT == DT_last:
        # each iteration fills in another missing value
        DT_last = DT
        DT = steps_2_thru_4(DT)
        
    # all missing values have been filled in original dataset
    return DT
    
def main(D, similarity_threshold=0.2):
    '''Main FIMUS algorithm.
    Returns copy of data with missing values imputed.
    D is 2D arr with cols being integer or categorical (ie strings).
        If col is floats, it should be prepared by 
        multiplying by power of 10 and rounding to int.
    A value of None in D signifies a missing value to be imputed.'''
    D_last = [row.copy() for row in D]
    D_curr = steps_2_thru_4(D_last, similarity_threshold)
    while not D_last == D_curr:
        D_last = D_curr
        D_curr = steps_2_thru_4(D_last, similarity_threshold)
    return D_curr

def steps_2_thru_4(D, similarity_threshold):
    '''Returns copy of D with one missing value imputed.'''
    B = missing(D)
    DG = generalise(D)
    domains = [domain(col) for col in cols(DG)]
    C = coappearances(DG)
    S = [similarity(DG, d, C) for d in domains]
    K = correlation(DG, C)
    
    for i in range(len(DG)):
        for j in range(len(DG[0])):
            if B[i][j] == 1:    # impute this missing value
                d = domains[j]
                # best vote
                x = CSR(DG[i],
                        j,
                        C,
                        S,
                        K,
                        similarity_threshold,
                        DG)

                if is_numeric(list(cols(D))[j]):
                    # another round with specific vals instead of generalised ranges
                    DN = records_in_range(x, # this category attr is a range
                                          j, # check this col
                                          D) # using original dataset
                    DN.insert(0, D[i])
                    C_new = coappearances(DN)
                    S_new = [similarity(DN, d, C_new) 
                             for d in (domain(col) for col in cols(DN))]
                    K_new = correlation(DN, C_new)
                    domain_new = domain(list(cols(DN))[j])

                    # new x, value within range of prev x attr
                    new_x = CSR(D[i],
                            j,
                            C_new,
                            S_new,
                            K_new,
                            similarity_threshold,
                            DN)
                    x = new_x
                # fill missing val in original using this best vote
                D_new = [row.copy() for row in D]
                D_new[i][j] = x
                return D_new
    return D

def missing(D):
    '''Returns matrix with 1 at all locs where D is None.'''
    B = [[1 if val == None else 0 
          for val in row]
         for row in D]
    return B    

def domain(col):
    '''Returns sorted list of unique values in col.
    Assumes col is categorized.'''
    seen = set()
    for val in filtered(col):
        if val not in seen:
            seen.add(val)
    return sorted(seen)
        
def bin_size(col):
    '''Categorization helper that determines size of ranges in numeric col.
    Used for FIMUS strategy of categorization/generalisation.'''
    col = filtered(col)
    return int((
        max(col) - min(col) + 1) ** 0.5) 

def zeros(n_rows, n_cols):
    return [[0 for col in range(n_cols)]
            for row in range(n_rows)]
    
def cols(D):
    '''Returns generator of cols in arr.'''
    return ([row[c] for row in D] 
            for c in range(len(D[0])))

def transpose(D):
    '''Returns copy of arr as col by rows instead of row by cols.'''
    return list(cols(D)) 

def is_numeric(val):
    '''Returns true if val is int or float, or is list containing those.
    Assumes if list[0] is numeric, then whole list is numeric.'''
    if type(val) == list:
        i = 0
        while i < len(val) and val[i] == None:
            i += 1
        return _is_val_numeric(val[i])
    else:
        return _is_val_numeric(val)

def _is_val_numeric(val):
    return type(val) == float or type(val) == int

def filtered(col):
    '''Returns list excluding None values.'''
    return [val for val in col if not val == None]
    
def generalise(D):
    '''Returns copy of D with numeric values turned into ranges.
    Categorizes numeric cols.
    The resulting matrix will then be all categorized.'''
    columns = []
    for col in cols(D):
        if not is_numeric(col):
            columns.append(col)
            continue

        size = bin_size(col)
        
        binned_col = []
        _min = min(filtered(col))

        for a in col:
            if a is None:
                binned_col.append(None)
                continue
                
            lower = _min + ( (a - _min) // size ) * size
            upper = lower + size - 1
            binned_col.append((lower, upper))
        columns.append(binned_col)
    return transpose(columns)
    
def coappearances(DG):
    '''Returns large matrix C containing 
    number of records with any two attribute values.
    Also called contingency table.'''
    domains = [domain(col) for col in cols(DG)]

    def is_matching(record, pattern):
        (j, Aj), (m, Am) = pattern
        return (record[j] == Aj and
                record[m] == Am)
                  
    # add index to domain, then combinate
    domain_combos = itertools.combinations(
        ((i, d) for i, d in enumerate(domains)), r=2)
    
    # just add index to attrs in each domain
    domain_combos = [[[(j, a) for a in Aj]
                       for (j, Aj) in domains]
                     for domains in domain_combos]
    
    # combinations bt attrs bt domain1, domain2
    combo_patterns = [itertools.product(d1, d2)
                      for d1, d2 in domain_combos]
    
    # all in one list, not separated per domain combo
    # pattern format: ((loc, val), (loc, val))
    all_patterns = [pattern
                    for domain_combo in combo_patterns
                    for pattern in domain_combo]
    
    size = sum((len(d) for d in domains))
    C = zeros(size, size)
    for record in DG:
        for pattern in all_patterns:
            if is_matching(record, pattern):
                r, c = location_of_pattern(pattern, domains)
                C[r][c] += 1
                C[c][r] += 1
    return C

def location_of_pattern(pattern, domains):
    '''Returns r,c of coapperance pattern in C.'''
    (j, Aj), (m, Am) = pattern
    return (location_of_attr(Aj, j, domains),
            location_of_attr(Am, m, domains))

def location_of_attr(val, domain_num, domains):
    '''Returns index of val relative to all domains, ie index on an axis of C.'''
    return (sum_prev_indices(domain_num, domains) + 
            domains[domain_num].index(val))

def sum_prev_indices(domain_num, domains):
    '''Returns sum of sizes of prev domains. 
    Used in location_of_attr() to help get index of a val in C.'''
    return sum((len(d) for d in domains[:domain_num]))

def similarity(DG, _domain, C):
    '''Returns similarity values between attributes in this domain.
    Seems to follow implementation at
    https://crpit.scem.westernsydney.edu.au/confpapers/CRPITV134Giggins.pdf,
    but results not quite same as in FIMUS paper.'''
    
    def degree(C_row):
        '''Returns number of times attr has coappearances,
        ie sum of non-zero values in a row of C.'''
        return sum(C_row)
    
    size = len(_domain)
    s = zeros(size, size)
    domains = [domain(col) for col in cols(DG)]
    
    pairs = itertools.combinations(_domain, r=2)
    for Ai, Aj in pairs:
        prev_indices = sum_prev_indices(domains.index(_domain), domains)
        i = _domain.index(Ai)
        j = _domain.index(Aj)
        Ci = C[prev_indices + i]
        Cj = C[prev_indices + j]

        numerator = sum(( (Ci[k] * Cj[k]) ** 0.5
                         for k in range(len(C))))
        denominator = ( degree(Ci) * degree(Cj) ) ** 0.5
        s_prime = numerator / denominator
        
        s[i][j] = s_prime
        s[j][i] = s_prime
        
    # fill reflective same-attr matches along main diagonal
    for i in range(size):
        s[i][i] = 1
        
    return s
        
def normalize(s):
    '''Returns proportional copy of similarity matrix where rows sum to 1.'''
    normalized = []
    for row in s:
        _sum = sum(row)
        normalized.append([val/_sum for val in row])
    return normalized

def correlation(DG, C):
    '''Returns table of correlations between each domain in DG.
    Uses Pearson's contingency coefficient that uses contingency table C.'''
    size = len(DG[0])
    K = zeros(size, size)
    
    C_exp = expected(C)
    
    domains = [domain(col) for col in cols(DG)]
    domain_combos = itertools.combinations(((i, d) for i, d in enumerate(domains)), r=2)
    for (i, domain_i), (j, domain_j) in domain_combos:
        
        r_start = sum_prev_indices(i, domains)
        r_end = r_start + len(domain_i)
        c_start = sum_prev_indices(j, domains)
        c_end = c_start + len(domain_j)
        
        X2 = 0  # chi squared
        N = 0   # total sample size of this section
        for r in range(r_start, r_end):
            for c in range(c_start, c_end):
                obs = C[r][c]
                exp = C_exp[r][c]
                X2 += (obs - exp)**2 / exp
                N += obs                

        pearsons_coefficient = ( X2 / (N + X2) ) ** 0.5
        K[i][j] = pearsons_coefficient
        K[j][i] = pearsons_coefficient
    return K

def expected(C):
    '''Returns expected distribution in values of C
    by using marginal distributions (ie sums of rows and cols).
    Used for chi-squared statistic in correlation().'''
    row_sums = [sum(row) for row in C]
    col_sums = [sum(col) for col in cols(C)]
    total = sum(row_sums)   # also == sum(col_sums)
    
    r_size = len(row_sums)
    c_size = len(col_sums)
    C_exp = [[0 for col in range(c_size)]
             for row in range(r_size)]
    coords = itertools.product(range(r_size), range(c_size))
    for i, j in coords:
        C_exp[i][j] = row_sums[i] * col_sums[j] / total
    
    return C_exp    
  
def CSR(record, j, C, S, K, similarity_threshold, DG):
    '''Returns best candidate to fill missing val at col j in record.
    Uses voting algorithm of FIMUS.'''
    
    domains = [domain(col) for col in cols(DG)]
    _domain = domain(list(cols(DG))[j])
    # vote format: (attr_val/candidate, vote_strength)
    votes = []
    for candidate in _domain:
        vote = 0
        for n, neighbor_col in enumerate(cols(DG)):
            neighbor_domain = domain(neighbor_col)
            neighbor = record[n]     # val of record in this column (horiz neighbor)
            
            if j == n or neighbor == None:
                continue    # skip when in same col or neighbor doesn't exist
                
            V_nx = 0                # vote only considering this attr value
            V_sx = 0                # vote considering all attr values in this domain
            
            c_candidate = location_of_attr(candidate, domains.index(_domain), domains)
            c_neighbor = location_of_attr(neighbor, n, domains)
            
            # add vote value considering actual neighbor at this col
            if similarity_threshold > 0:
                V_nx = ( C[c_candidate][c_neighbor] / 
                        frequency(neighbor, neighbor_col) )
                
            if similarity_threshold < 1:
                neighbor_siblings = neighbor_domain.copy()
                neighbor_siblings.remove(neighbor)
                # add vote contribution for all other possible values of neighbor at this col
                for neighbor_sibling in neighbor_siblings:   # other vals in domain of this column
                    c_neighbor_sibling = location_of_attr(neighbor_sibling, n, domains)
                    H = ( C[c_candidate][c_neighbor_sibling] / 
                         frequency(neighbor_sibling, neighbor_col) )
                    s_neighbor = neighbor_domain.index(neighbor)
                    s_neighbor_sibling = neighbor_domain.index(neighbor_sibling)
                    # multiply vote contribution by similarity score
                    V_sx += H * S[n][s_neighbor][s_neighbor_sibling]
                    
            k_domain = domains.index(_domain)
            k_neighbor_domain = domains.index(neighbor_domain)
            # weighted vote contribution from this neighbor col
            V_px = (V_nx * similarity_threshold + 
                    V_sx * (1 - similarity_threshold)) * K[k_domain][k_neighbor_domain]
            vote += V_px
        votes.append((candidate, vote))
            
    best_candidate = max(votes, key=lambda x:x[1])[0]
    return best_candidate
    
def frequency(attr, col):
    '''Returns total num of appearances of value in list.
    Used in CSR vote calculation.'''
    return sum((True if val_at_row == attr else False
                for val_at_row in col))

def records_in_range(_range, col_num, data):
    '''Returns records in original data where vals in col fall in range.
    Range is inclusive tuple (lower, upper). 
        Floats would not work well, 
        eg gap between bins/ranges (10.0, 14.0), (15.0, 19.0).
    Used to generate new D in transition to deeper CSR.'''
    _min, _max = _range
    return [record for record in data
            if (not record[col_num] == None and
                record[col_num] >= _min and 
                record[col_num] <= _max)]