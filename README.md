This example contains a draft of a new extrusion simulation code. 

## Code architecture and assumption 

### Entities 

* **LEFs** - a (NLEFs x 2) array of positions of two LEF legs 
* **statuses** - a (NLEFs x 2) array of statuses (moving, paused, bound, etc.) of each LEF leg




### Loading of LEFs to the sequence 

We assume random loading of a LEF. To load, a LEF needs a width-3 open area. LEF either loads to positions [-1,0] or positions [-1,1] of the area 

As such, it is advisable to set loading probabilities of all CTCF sites to zero - so you can't load "across" a CTCF site. 
Also, extreme care needs to be taken with respect to what does it mean to "load" at enhancers. 
E.g. in cases with one-sided extrusion you may want one of the "ends" to remain at the enhancer while another to translocate away. 
In that case, you should modify the loading function carefully - e.g. you may forgo the width-2 loading at all. 

Width-2 loading is needed to avoid checkering in "naive" extrusion. 
Checkering would happen because if you load a LEF of width 1, only width 1, 3, 5, 7... would be possible until one of the two legs gets stopped at something, 
because we increase the LEF by 2 every step. 
Adding uniform pause probability of .25 will get rid of that checkering after 10-ish steps anyways, so maybe that is a solution. 
(in which case I would advice to load at (-1, 0) or (0,1) to be symmetric)
But for now, we are fighting checkering through width-2 loading, which would sometimes load "around" the loading site, but that's OK. 

### Unloading of LEFs from the sequence 

Each LEF leg will have a status (bound, moving, stalled) - and will have a corresponding Unload probability. 

The question is how to combine two legs' unloading probability into unloading for the whole LEF? 

The solution was made to take an average except when one of the two legs is stuck at CTCF ("bound"). 
When the leg is stuck at CTCF, it has a protective effect on the whole cohesin, including the other leg. 
So in that case we take the smallest unloadig 

#### simple competition 
**one leg is at a site with higher unloading probability** - probably should select the higher probability for the whole LEF  (can ignore it for now).


#### Unloding of stalled (not bound by CTCF, just stalled) cohesins

The idea behind it is that "moving" protects a LEF from being unloaded. As soon as it is stalled at something, it can get unloaded. 
A LEF is stalled against something 
"Stalled" status would get a higher unloading rate. 
