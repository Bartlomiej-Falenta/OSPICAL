# Optomod,
Or: all-optical modulation and amplification component.

The idea for component came from noticing that to achieve all-optical computing in foreseeable future, some method of amplifying and modulating or switching signals with gain factor above 2 is absolutely needed. So, after some amount of initial reseacrh I've started making some simulations, which will be uploaded to this directory soon. They are mostly done in Python, sometimes C++ or Lumerical Scripting Format + LaTeX for formulas present in this directory – also, some drawing and sketches. The general principle is combination of multiple effects and mechanisms:
 - Optical Kerr Effect, in this case, requiring VERY high nonlinearities achieved with properly combined materials), which introduces functions like switching, modulation, and „clean-up” of a signal;
 - Stimulated Raman Scattering (Brillouin's can be used as well – both SRS abd SBS need special, possibly organic materials), which actually amplifies the signal, with possibility of amplification by few orders of magnitude (though we aim at about 50 to 500);
 - Mach-Zehnder Interferometer (in this specific case, Mach Zehnder Modulator) which converts OKE phase shifts to amplitude shift.

 [TO DO] : Possible material choices, formulas, diagrams, simulation scripts and example results.
