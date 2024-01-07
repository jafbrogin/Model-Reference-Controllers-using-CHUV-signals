This code was developed for the thesis entitled: "A contribution to the dynamics of epilepsy: identification and control for seizure attenuation". It consists of a new approach to reconstructing and attenuating real epileptiform activity artificially based on model reference controllers. The signals used in these codes are not yet available on any platform because there is a restriction to their use inside the facilities of the Centre hospitalier universitaire vaudois (CHUV) in Lausanne, Swizterland. Considering this, the codes cannot be run yet, but they are already available on GitHub for transparency purposes. The article containing the results obtained using this code are about to be submitted and shall be available online soon. 

Essentially, the code was developed on the Spyder platform and it consists six separate files:
seizures_CHUV.py, which carries out a statistical analysis considering Power Spectral Density (PSD) and the Hilbert Transform (HT) based on the real seizures; ID_seizures_CHUV_OP.py, which imports the real signals and apply a grey-box system identification approach considering the Fourier basis for obtaining candidate models that represent the main oscillation behavior of the seizures (these models are obtained in open loop, but reconstruct the seizures in closed-loop similation); LMI_seizures_CHUV_simple.py, which is an optimization routine for obtaining the gains of the controllers for seizure reconstruction and attenuation; rk4_seizures_CHUV_adaptive_full.py and rk4_seizures_CHUV_simple.py, which perform the closed-loop simulation of the candidate models for seizure reconstruction and attenuation considerign adaptive and fixed gains, respectively; and, at last, PCA_CHUV_MRC.py, which analyzes the PSDs of the signals using Principal Component Analysis to assess if seizure and non-seizure behavior can be spatially separated.

For further information or questions, please contact: ferres.brogin@unesp.br or ferres.brogin@gmail.com.
