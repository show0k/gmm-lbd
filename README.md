# GMM-lbd
Gaussian mixture experiments for learning by demonstration

Notebook examples:
* [example 1](https://github.com/show0k/gmm-lbd/blob/master/notebooks/calinon_tests.ipynb)
* [example 2](https://github.com/show0k/gmm-lbd/blob/master/notebooks/working_demo.ipynb) 

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



#### Pypot records improvements (TODO)
* record datas with a variable framerate (compression + CPU usage for generating GMMs )