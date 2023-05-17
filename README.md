# fimus.py

A data pre-processing algorithm that predicts values for correlated numerical and categorical records.

Implemented from a research paper into python.

## Usage

```python
import fimus

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

D_imputed = fimus.main(D)
```

Or run `python test_fimus.py`

## Dependencies

Tested on `python 3.7`

## Notes

- I couldn't quite replicate the results from the paper; the numerical similarity calculation in step 3 is where tests fail.

- Performing on real-world data like `yeast.data` had even more error, including with the categorical data.

- Fun fact / background: some stranger in Upwork (freelancing) requested a python implementation of this algorithm. The task hooked my curiosity and I wanted to prove my skills to myself, so I gave it a shot despite not getting the job.

- I've dug up the code in case someone else finds it useful. It's some of my early programming and a little messy, and I'm not sure how to fix the similarity calculations. Oh well!

- The original FIMUS algorithm was implemented by the students/researchers in Java.

- Relevant papers:
	- https://www.researchgate.net/publication/259185364_FIMUS_A_Framework_for_Imputing_Missing_Values_Using_Co-appearance_Correlation_and_Similarity_Analysis
	- https://github.com/ipranavpatel/ExtendedFIMUS/blob/master/Report.pdf
	- https://crpit.scem.westernsydney.edu.au/confpapers/CRPITV134Giggins.pdf
