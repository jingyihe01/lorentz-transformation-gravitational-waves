# Lorentz Transformation of Gravitational-Wave Polarizations

This repository contains a Python implementation of the Lorentz transformation of a three-dimensional gravitational-wave tensor, based on the framework developed by He, Liu, and Cao (2023).

The code transforms the source-frame gravitational-wave polarizations <i>h</i><sub>+</sub>(<i>t</i>) and <i>h</i><sub>×</sub>(<i>t</i>) into the detector-frame polarizations <i>H</i>′<sub>+</sub>(<i>t</i>′) and <i>H</i>′<sub>×</sub>(<i>t</i>′) for a source moving with an arbitrary relativistic velocity.

## Overview

The implementation includes:

- Construction of the polarization tensors <i>e</i><sup>+</sup><sub>ij</sub> and <i>e</i><sup>×</sup><sub>ij</sub>
- Generation of a circular-binary waveform using the quadrupole approximation
- Construction of the three-dimensional transverse-traceless gravitational-wave tensor <i>h</i><sub>ij</sub>
- Exact Lorentz transformation from <i>h</i><sub>ij</sub> to <i>h</i>′<sub>ij</sub>
- Relativistic aberration of the gravitational-wave propagation direction
- Rotation between the source and detector polarization bases
- Transformation of the time coordinate using the Doppler factor
- Projection of the transformed tensor onto the detector-frame polarization basis
- Numerical verification of the transverse and traceless properties
- Visualization of the waveforms before and after the Lorentz boost

The complete transformation pipeline is:

<p align="center">
<i>h</i><sub>+</sub>, <i>h</i><sub>×</sub>
&nbsp;→&nbsp;
<i>H</i><sub>+</sub>, <i>H</i><sub>×</sub>
&nbsp;→&nbsp;
<i>H</i><sub>ij</sub>
&nbsp;→&nbsp;
<i>H</i>′<sub>ij</sub>
&nbsp;→&nbsp;
<i>H</i>′<sub>+</sub>, <i>H</i>′<sub>×</sub>
</p>

The implementation mainly follows Eqs. (50), (59), (79), (121)–(122), (126)–(132), and (142)–(143) of the reference paper.

## Code Structure

The program contains the following components:

1. **Polarization bases**

   Constructs the spherical basis vectors and the plus and cross polarization tensors.

2. **Circular-binary waveform**

   Generates a simple source-frame waveform using the quadrupole approximation.

3. **Tensor construction and projection**

   Converts between the two polarization components and the three-dimensional gravitational-wave tensor.

4. **Lorentz transformation**

   Applies the exact transformation of the gravitational-wave tensor given by Eq. (59) of the reference paper.

5. **Relativistic aberration**

   Calculates the transformed propagation direction using Eq. (79).

6. **Coordinate transformation**

   Rotates the velocity components between the source and detector coordinate systems.

7. **Detector-frame polarizations**

   Projects the transformed tensor onto the new polarization basis to obtain <i>H</i>′<sub>+</sub> and <i>H</i>′<sub>×</sub>.

8. **Numerical demonstration**

   Generates example waveforms, performs consistency checks, and plots the results.

## Main Function

The principal function is:

```python
source_h_to_detector_Hprime(
    hplus,
    hcross,
    iota,
    Theta,
    Phi,
    Psi,
    beta_XYZ,
)
```

### Inputs

- `hplus`: source-frame plus polarization, <i>h</i><sub>+</sub>(<i>t</i>)
- `hcross`: source-frame cross polarization, <i>h</i><sub>×</sub>(<i>t</i>)
- `iota`: binary inclination angle, ι
- `Theta`: source polar angle, Θ
- `Phi`: source azimuthal angle, Φ
- `Psi`: polarization angle, Ψ
- `beta_XYZ`: dimensionless source velocity vector, (v<sub>X</sub>, v<sub>Y</sub>, v<sub>Z</sub>)/c

All angular inputs are expressed in radians.

The boost velocity must satisfy:

<p align="center">|<b>β</b>| &lt; 1</p>

### Outputs

The function returns a Python dictionary containing:

- `Hplus`: detector-basis plus polarization before the boost
- `Hcross`: detector-basis cross polarization before the boost
- `Hplus_prime`: transformed polarization <i>H</i>′<sub>+</sub>
- `Hcross_prime`: transformed polarization <i>H</i>′<sub>×</sub>
- `Theta_prime`: transformed polar angle Θ′
- `Phi_prime`: transformed azimuthal angle Φ′
- `k`: time-transformation factor
- `Hij`: gravitational-wave tensor before the boost
- `Hij_prime`: gravitational-wave tensor after the boost
- `Rhat_prime`: transformed source-direction vector

For a source moving at constant velocity, the source-frame and detector-frame times are related by:

<p align="center"><i>t</i>′ = <i>k t</i></p>

## Requirements

- Python 3
- NumPy
- Matplotlib

Install the required packages using:

```bash
pip install numpy matplotlib
```

## Usage

Save the program as:

```text
lorentz_gw_polarization.py
```

Run the numerical demonstration using:

```bash
python lorentz_gw_polarization.py
```

The demonstration performs the following steps:

1. Generates <i>h</i><sub>+</sub>(<i>t</i>) and <i>h</i><sub>×</sub>(<i>t</i>) for a circular binary.
2. Rotates the waveform into the detector polarization basis.
3. Constructs the three-dimensional tensor <i>H</i><sub>ij</sub>.
4. Applies the exact Lorentz transformation.
5. Calculates <i>H</i>′<sub>+</sub>(<i>t</i>′) and <i>H</i>′<sub>×</sub>(<i>t</i>′).
6. Checks that the transformed tensor remains transverse and traceless.
7. Produces plots comparing the original and transformed waveforms.

The physical parameters in `run_demo()` are illustrative. They can be replaced with the desired component masses, orbital frequency, luminosity distance, source orientation, and velocity.

## Numerical Checks

The program evaluates:

- The trace of the transformed tensor
- The contraction between the transformed tensor and the aberrated propagation direction

For a valid transverse-traceless tensor, both quantities should be close to zero within numerical precision.

## Reference

X. He, X. Liu, and Z. Cao, “Lorentz transformation of three dimensional gravitational wave tensor,” arXiv:2302.07532 (2023).

Paper: https://arxiv.org/abs/2302.07532

## Notice

This code was developed as part of a summer research project and is provided for viewing and academic reference only.

Redistribution or modification requires prior permission.
https://arxiv.org/abs/2302.07532


