# Exploration-of-LCS-on-CVA

Welcome to Exploration of Lagrangian Coherent Structures on Crowd Video Analysis! :blush:

This is where you can find the code for all of the programs used in the making of my project and the final project report. We give a brief summary of the idea of the project below:

> The idea is to take a video of a moving crowd (pedestrians/vehicles) and use a metric called the Finite Time Lyapunov Exponent Field (FTLE Field) to determine the regions where crowd flows are most turbulent or contrasting. More specifically, the FTLE field gives a high output if the flows of crowds are diverging from each other and a low output if the flows are converging/stable to each other. In practise, we apply (Farneback) optical flow to the video to obtain it's velocity vectors and then calculate the FTLE field. The FTLE field has been used in many research papers for a myriad of applications, this is one of those where the ultimate goal was the detection of the forming of a dangerous crowd that could risk the lives of pedestrians. 

### File Descriptions:

**Exploration of LCS on CVA.pdf** - Final report document detailing the exact process and findings of the project.

**FTLE.py** - Calculates the FTLE field for a given analytical velocity field function

**FTLE_CVA.py** - Calculates the FTLE field for a given video file.

**Farneback.py** - Calculates the Optical flowfield using Farenback Optical Flow.

**Perturbation_Direc.py** - Calculate the Tangent Repulsion Rate, Normal Repulsion Rate and Repulsion Ratio for a given video
and output the result as a scalar field plot.

**plotParticles.py** - For a given analytical velocity field output a video of a Grid or Blob of particles being advected by that velocity field. 


If you have any questions please contact - bryantiger225@gmail.com
