[![Binder](http://mybinder.org/badge.svg)](http://mybinder.org/repo/show0k/gmm-lbd) 

# GMM-lbd
Gaussian mixture experiments for learning by demonstration

Notebook examples:
* [example 1](notebooks/calinon_tests.ipynb)
* [example 2](notebooks/working_demo.ipynb) 

## Installation
```bash
pip install gmm-lbd
```

OR if you clone this repository, you can make a 

```bash
python setup.py develop
```

## Roadmap
### Low level (Gaussian mixture)
* [X] Automaticly choose the number of gaussians
* [x] conditional probability of GMM
* [x] regression of GMM
* [X] product of retrived means and covariances of GMM
* [ ] product with non consistent shapes
* [ ] speed management 

### High level (combinaison of movements)
* [X] Quick add pypot records for any motors
* [X] Easyly represent GMM with ellipses
* [ ] Plot ellipse for GMM with more than 2 dimensions
* [ ] Align movements with DTW  
* [ ] White detection (at begin and end of movements)
* [X] Sequential : concatenation of GMMs
* [X] Concurent : product 
* [X] Add a coefficiant to rise or low the importance of a movement
* [ ] Add a filter in the sequential combinaison
* [ ] Adapt to use an IK model, for performing task space trajectory: __WANTED__
* [ ] Incremental definition of the GMM for each new representation @Calinon07HRI (very good idea for online learning)



#### Pypot records improvements (TODO)
* record datas with a variable framerate (compression + CPU usage for generating GMMs )
