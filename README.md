This example contains a draft of a new extrusion simulation code. 

## Code architecture and assumption 

### Glossary 

#### Entities

* **LEF**:  that thing that extrudes loops. Cohesin in most cases. 
* **leg**: a "virtual leg" of cohesin that extrudes on one side. Whether or not cohesin has legs (or other body parts), if there is a loop from A to B, cohesin's "legs" are left at position A and right at position B 

#### processes 

* **loading**: a process of cohesin appearing on chromatin, by whatever means - we don't differentiate between targeted loading (e.g. at enhancers) and generic loading/binding/etc. anywhere along the sequence. 
* **unloading**: the opposite of loading. We don't differentiate active unloading and passive unloading. 
* **capture**: the process of cohesin becoming "captured" at a CTCF site and becoming a CTCF-bound (and possibly CTCF-stabilized) cohesin 
* **release**: the opposite of capture - cohesin's leg is released from CTCF and now is a normal moving leg. 

#### Conventions and internal variables

* **status**: one of ["moving", "paused", "captured"]   - a status of a cohesin leg. 
  * **moving** - means the leg was just created, just released, or previously successfully moved 
  * **paused** - means that the leg failed to move, because it got randomly paused 
  * **stalled** - means that the leg failed to move because something is in the way
    * A leg is always stalled if it is not captured and something is in the way 
  * **captured** - means that the leg is bound to CTCF and cannot move


#### Arrays 

* **LEFs** - a (NLEFs x 2) array of positions of two LEF legs 
* **statuses** - a (NLEFs x 4) array of statuses (moving, paused, stalled, captured) of each LEF leg

### Loading of LEFs to the sequence 

We assume random loading of a LEF with some load probability as a function of sequence.
Loading always happens upon unloading, so the probabilities are summed and re-normalized to sum to 1.
Note that actual loading will happen with a different distributions as some sites will be occupied more often than others.
To load, a LEF needs a width-3 open area. LEF either loads to positions [-1,0] or positions [-1,1] of the area 

As such, it is advisable to set loading probabilities of all CTCF sites to zero - so you can't load "across" a CTCF site. 
Also, extreme care needs to be taken with respect to what does it mean to "load" at enhancers. 
E.g. in cases with one-sided extrusion you may want one of the "ends" to remain at the enhancer while another to translocate away. 
In that case, you should modify the loading function carefully - e.g. you may forgo the width-2 loading at all. 

Width-2 loading is needed to avoid **checkering** in "naive" extrusion. 
Checkering would happen because if you load a LEF of width 1, only width 1, 3, 5, 7... would be possible until one of the two legs gets stopped at something, 
because we increase the LEF by 2 every step. 
Adding uniform pause probability of .25 will get rid of that checkering after 10-ish steps anyways, so maybe that is a solution. 
(in which case I would advice to load at (-1, 0) or (0,1) to be symmetric)
But for now, we are fighting checkering through width-2 loading, which would sometimes load "around" the loading site or a CTCF, but that's OK. 
We will never load around another LEF's leg as there is a check for that. 

### Unloading of LEFs from the sequence 

Each LEF leg will have a status (bound, moving, stalled) - and will have a corresponding Unload probability. 

The question is how to combine two legs' unloading probability into unloading for the whole LEF? 

The solution was made to take an average except when one of the two legs is stuck at CTCF ("bound"). 
When the leg is stuck at CTCF, it has a protective effect on the whole cohesin, including the other leg. 
So in that case we take the smallest unloadig probability. 
