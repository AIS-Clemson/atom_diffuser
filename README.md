# AtomDiffuser: Time-Aware Degradation Modeling for Drift and Beam Damage in STEM Imaging

Our paper is accepted by 
<a href="https://sites.google.com/view/cv4ms-iccv-2025" target="_blank">ICCV 2025 CV4MS Workshop</a>

See our paper at the ICCV 2025: 
<a href="./exp/Poster_hao9_ICCV_2025.pdf" target="_blank">ICCV 2025 Poster</a> 
/ 
<a href="https://openaccess.thecvf.com/content/ICCV2025W/CV4MS/html/Wang_AtomDiffuser_Time-Aware_Degradation_Modeling_for_Drift_and_Beam_Damage_in_ICCVW_2025_paper.html" target="_blank">Paper</a>


See our project page for the practical implementation and testing:
<a href="https://github.com/AIS-Clemson/atom_diffuser" target="_blank">AIS-Clemson/AtomDiffuser</a>
 
## Project Overview
Scanning transmission electron microscopy (STEM) plays a critical role in modern materials science, enabling direct imaging of atomic structures and their evolution under external interferences. However, interpreting time-resolved STEM data remains challenging due to two entangled degradation effects: spatial drift caused by mechanical and thermal instabilities, and beam-induced signal loss resulting from radiation damage. These factors distort both geometry and intensity in complex, temporally correlated ways, making it difficult for existing methods to explicitly separate their effects or model material dynamics at atomic resolution.

<div align="center">
    <img src="./exp/cover.svg" alt="" style="width: 50%;">
</div>

In this work, we present AtomDiffuser, a time-aware degradation modeling framework that disentangles sample drift and radiometric attenuation by predicting an affine transformation and a spatially varying decay map between any two STEM frames. Unlike traditional denoising or registration pipelines, our method leverages degradation as a physically heuristic, temporally conditioned process, enabling interpretable structural evolutions across time. Trained on synthetic degradation processes, AtomDiffuser also generalizes well to real-world cryo-STEM data. It further supports high-resolution degradation inference and drift alignment, offering tools for visualizing and quantifying degradation patterns that correlate with radiation-induced atomic instabilities.
<br> 

<div align="center">
    <img src="./exp/frame.svg" alt="" style="width: 100%;">
</div>

 
### Acknowledgements:
Please cite our work if you find this project helpful.
```bibtex
@InProceedings{Wang_2025_ICCV,
    author    = {Wang, Hao and Zheng, Hongkui and He, Kai and Razi, Abolfazl},
    title     = {AtomDiffuser: Time-Aware Degradation Modeling for Drift and Beam Damage in STEM Imaging},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops},
    month     = {October},
    year      = {2025},
    pages     = {3631-3640}
}
```

<br>
<br>
<br>



<br>
<br>
