# *Foraging on Autopilot*: Fundamental Visual Navigation Algorithms without Distance or Prediction

<img src="./site_media/flow.png" width="800"/>

**Author of Model:** Patrick Govoni <br>
**Supervisors:** Prof. Pawel Romanczuk <br>
**Affiliation:** Institute for Theoretical Biology, Department of Biology, Humboldt Universität zu Berlin <br>
**Group:** [Collective Information Processing Lab](http://lab.romanczuk.de/) <br>
**Timespan:** 2023-Present

**Abstract:** <br>
Foraging in predictable environments requires coordinating effective movement with observable
spatial context, i.e. navigation. Separate from search, navigation is controlled by two partially
dissociable, concurrently developed systems in the brain. The cognitive map informs an organism
of its location, bearing, and distances between environmental features, enabling shortcuts. Visual
response-based navigation via routes, on the other hand, is commonly considered inflexible, ultimately
subserving map-based representations. As such, navigation models widely assume the primacy of
maps, constructed through predictive control and distance perception, while neglecting response-
based strategies. Here we show the sufficiency of a minimal feedforward framework in a classic
navigation task. Our agents, directly translating visual perception to movement, navigate to a hidden
goal in an open field, an environment often assumed to require predictive map-based representations.
While visual distance enables direct trajectories to the goal, two distinct algorithms develop to
robustly navigate using visual angles alone. Each of the three confers unique tradeoffs as well as
aligns with movement behavior observed in rodents, insects, fish, and sperm cells, suggesting the
broad significance of response-based navigation throughout biology. We advocate further study of
bottom-up, response-based navigation without assuming online access to computationally expensive
distance perception, prediction, or maps, which may better explain behavior under energetic or
attentional constraints.


<p float="left">
  <img src="./site_media/sim_IS_respawn.gif" width="250" />
  <img src="./site_media/sim_BD_respawn.gif" width="250" />
  <img src="./site_media/sim_DP_respawn.gif" width="250" />
</p>
<p float="left">
  <img src="./site_media/trajs_IS.png" width="250" />
  <img src="./site_media/trajs_BD.png" width="250" />
  <img src="./site_media/trajs_DP.png" width="250" />
</p>

<img src="./site_media/convergence.png" width="800"/>

**Manuscript:** <br>
[Preprint](https://arxiv.org/abs/2407.13535) <br>

**Citation:** <br>
Govoni, P., Romanczuk, P. Fundamental Visual Navigation Algorithms: Indirect Sequential, Biased Diffusive, & Direct Pathing. (2024). 

**License:** <br>
Copyright © 2023 [Patrick Govoni](https://github.com/pgovoni21). <br>
This project is [MIT](https://github.com/pgovoni21/vis-nav-abm?tab=MIT-1-ov-file) licensed.
