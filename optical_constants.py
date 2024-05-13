"""
Various parametrization of the optical constants.
==================================================

Analytical expressions are taken from [1].

References
----------
   [1] A. S. Ferlauto, G. M. Ferreira, J. M. Pearce, C. R. Wronski, R. W. Collins, X. Deng, and G. Ganguly,
       J. Appl. Phys. 92, 2424 (2002).
       https://doi.org/10.1063/1.1497462
   [2] M. Fox Optical Properties of Solids. 2nd Ed. (2010)
"""
import cmath
import math

from scipy.special import expi as Ei

__version__ = '0.1'


class Dispersion:
    """Base class for various dispersion models.
    """

    def __init__(self, eps_inf=1):
        self.eps_inf = eps_inf

    def eps_1(self, E) -> float:
        # to be overridden
        return self.eps_inf

    def eps_2(self, E) -> float:
        # to be overridden
        return 0

    def n(self, E):
        """
        Index of refraction.
        """
        e1 = self.eps_1(E)
        e2 = self.eps_2(E)
        return math.sqrt((math.hypot(e1, e2) + e1) / 2)

    def k(self, E):
        """
        Extinction coefficient.
        """
        e1 = self.eps_1(E)
        e2 = self.eps_2(E)
        return math.sqrt((math.hypot(e1, e2) - e1) / 2)

    def R(self, E):
        """
        Reflectivity.
        """
        n = self.n(E)
        k = self.k(E)
        return ((n - 1) ** 2 + k ** 2) / ((n + 1) ** 2 + k ** 2)

    def a(self, E):
        """
        Absorption coefficient.
        """
        k = self.k(E)
        wavelength = 1239.841973862093e-09 / E
        return 4 * math.pi * k / wavelength


class Sellmeier(Dispersion):
    """
    Sellmeier equation model for transparent media.
    """

    def __init__(self, B1, B2, B3, C1, C2, C3):
        Dispersion.__init__(self)
        self.B1 = B1
        self.B2 = B2
        self.B3 = B3
        self.C1 = C1
        self.C2 = C2
        self.C3 = C3

    def eps_1(self, E) -> float:
        w = 1.239841973862093 / E  # wavelength in um (provided E is in eV)
        w_sq = w ** 2
        return 1 + self.B1 * w_sq / (w_sq - self.C1) + \
            self.B2 * w_sq / (w_sq - self.C2) + \
            self.B3 * w_sq / (w_sq - self.C3)

    def eps_2(self, E) -> float:
        return 0


class JellisonModine(Dispersion):
    """G. Jellison and F. Modine results for Tauc-Lorentz parametrization of optical constants."""

    def __init__(self, A, E0, C, Eg, eps_inf=1):
        """
        :param A: Oscillator amplitude.
        :param E0: Resonance energy.
        :param C: Oscillator width.
        :param Eg: Band gap.
        """
        Dispersion.__init__(self, eps_inf)
        self.A = A
        self.E0 = E0
        self.C = C
        self.Eg = Eg
        self._E0sq = self.E0 ** 2
        self._Csq = self.C ** 2
        self._Egsq = self.Eg ** 2

    def eps_1(self, E):
        A, E0, E0sq, C, Csq, Eg, Egsq, pi, alpha, gamma2, zeta4, aln, aatan = self.A, self.E0, self._E0sq, self.C, \
            self._Csq, self.Eg, self._Egsq, math.pi, self._alpha, self._gamma2, self._zeta4(E), self._aln(E), \
            self._aatan(E)

        if self.E0 <= self.C / 2:  # critically- or over-damped DHO is not supported
            raise Exception(f'E0 must be greater than C/2')

        out = 0

        out += C * aln / (2 * alpha * E0) * \
               cmath.log((E0sq + Egsq + alpha * Eg) / (E0sq + Egsq - alpha * Eg))
        out -= aatan / E0 * \
               (pi - cmath.atan((2 * Eg + alpha) / C) + cmath.atan((-2 * Eg + alpha) / C))
        out += 2 * E0 / alpha * Eg * (E ** 2 - gamma2) * \
               (pi + 2 * cmath.atan(2 * (gamma2 - Egsq) / (alpha * C)))
        out -= E0 * C * (E ** 2 + Egsq) / E * \
               cmath.log(abs(E - Eg) / (E + Eg))
        out += 2 * E0 * C * Eg * \
               cmath.log(abs(E - Eg) * (E + Eg) / cmath.sqrt((E0sq - Egsq) ** 2 + Egsq * Csq))

        out *= A / (pi * zeta4)

        out += self.eps_inf

        if out.imag == 0:
            return out.real
        else:
            raise Exception(f'Non-zero imaginary part obtained for {E} eV: {out}')

    def eps_2(self, E):
        if E <= self.Eg:
            return 0
        else:
            return self.A * self.E0 * self.C * (E - self.Eg) ** 2 / (
                    (E ** 2 - self.E0 ** 2) ** 2 + self.C ** 2 * E ** 2) / E

    @property
    def _alpha(self):
        return cmath.sqrt(4 * self._E0sq - self._Csq)

    @property
    def _alpha2(self):
        return 4 * self._E0sq - self.C ** 2

    @property
    def _gamma2(self):
        return self._E0sq - self._Csq / 2

    def _zeta4(self, E):
        return (E ** 2 - self._gamma2) ** 2 + 0.25 * self._alpha2 * self.C ** 2

    def _aln(self, E):
        return (self._Egsq - self._E0sq) * E ** 2 + self._Egsq * self._Csq - self._E0sq * (self._E0sq + 3 * self._Egsq)

    def _aatan(self, E):
        return (E ** 2 - self._E0sq) * (self._E0sq + self._Egsq) + self._Egsq * self._Csq


class Foldyna(Dispersion):
    """
    From "Urbach tail extension of Tauc-Lorentz model dielectric function" by M. Foldyna.
    """

    def __init__(self, A, E0, C, Eg, Ec, eps_inf=1):
        """

        :param A: Oscillator amplitude (eV).
        :param E0: Resonance energy (eV).
        :param C: Oscillator width (eV).
        :param Eg: Band gap (eV).
        :param Ec: parameter describing Urbach tail width (eV).
        :param eps_inf: Real part of dielectric function at infinite frequency.
        """
        Dispersion.__init__(self, eps_inf)
        if E0 <= C / 2:  # critically- or over-damped DHO is not supported
            raise Exception(f'E0 must be greater than C/2')
        self._A = A
        self._E0 = E0
        self._E0sq = E0 ** 2
        self._C = C
        self._AE0C = A * E0 * C
        self._Csq = C ** 2
        self._gamma2 = self._E0sq - self._Csq / 2
        self._alpha = cmath.sqrt(4 * self._E0sq - self._Csq)
        self._Eg = Eg
        self._Egsq = Eg**2
        self._EgsqCsq = self._Egsq * self._Csq
        self._Ec = Ec
        self._Ecsq = Ec ** 2
        self._CsqEcsq = C ** 2 * Ec ** 2
        self._EcmEg = self._Ec - self._Eg
        self._EcsqmE0sq = self._Ecsq - self._E0sq
        c1 = self._CsqEcsq + self._EcsqmE0sq ** 2
        self._Eu = self._EcmEg / (
                3 - 2 * self._Ec * self._EcmEg * (self._Csq + 2 * self._EcsqmE0sq) / c1
        )
        self._Au = math.exp(-self._Ec / self._Eu) * (self._AE0C * self._EcmEg ** 2) / c1

    def _aL(self, E):
        return (self._Egsq - self._E0sq) * E**2 + self._EgsqCsq - self._E0sq * (self._E0sq + 3 * self._Egsq)

    def _aA(self, E):
        return (E**2 - self._E0sq) * (self._E0sq + self._Egsq) + self._EgsqCsq

    def _zeta4(self, E):
        Esq = E**2
        return (Esq - self._E0sq)**2 + self._Csq * Esq

    def _eps_1_TL(self, E):
        z4 = self._zeta4(E)
        absEcmE = abs(self._Ec - E)
        out = 0
        out -= self._AE0C * (E**2 + self._Egsq) / (math.pi * z4 * E) * cmath.log(absEcmE / (self._Ec + E))
        out += 2 * self._AE0C * self._Eg / (math.pi * z4) * \
               cmath.log(absEcmE * (self._Ec + E) / cmath.sqrt(self._EcsqmE0sq**2 + self._CsqEcsq))
        out += self._A * self._C * self._aL(E) / (2 * math.pi * z4 * self._alpha * self._E0) * \
               cmath.log((self._E0sq + self._Ecsq + self._alpha * self._Ec) /
                         (self._E0sq + self._Ecsq - self._alpha * self._Ec))
        out -= self._A * self._aA(E) / (math.pi * z4 * self._E0) * \
               (math.pi -
                cmath.atan((2 * self._Ec + self._alpha) / self._C) -
                cmath.atan((2 * self._Ec - self._alpha) / self._C))
        out += 4 * self._A * self._E0 * self._Eg * (E**2 - self._gamma2) / (math.pi * z4 * self._alpha) * \
               (math.pi / 2 - cmath.atan(2 * (self._Ecsq - self._gamma2) / (self._alpha * self._C)))

        if out.imag == 0:
            return out.real
        else:
            raise Exception(f'Non-zero imaginary part obtained for {E} eV: {out}')

    def _eps_1_U(self, E):
        EdEu = E / self._Eu
        return self._Au / (E * math.pi) * (math.exp(-EdEu) * (Ei(E / self._Eu) - Ei((self._Ec + E) / self._Eu)) +
                                           math.exp(EdEu) * (Ei((self._Ec - E) / self._Eu) - Ei(-EdEu)))

    def eps_1(self, E):
        return self.eps_inf + self._eps_1_TL(E) + self._eps_1_U(E)

    def eps_2(self, E):
        Esq = E ** 2
        if E > self._Ec:
            return 1 / E * (self._AE0C * (E - self._Eg) ** 2) / (
                        (Esq - self._E0sq) ** 2 + self._Csq * Esq)
        else:
            return self._Au / E * math.exp(E / self._Eu)


class Lorentz(Dispersion):
    """The Lorentz oscillator model considers the interaction between a light
    wave and an atom with a single resonance frequency due to bound
    electrons.
    In terms of DHO, oscillator could be under-damped (E0 > Gamma/2), critically damped (E0 = Gamma/2) or over-damped
    (E0 < Gamma/2). For under-damped DHO, the resonant energy could be defined as Er = sqrt(E0^2 - Gamma^2 / 4).
    """

    def __init__(self, A, E0, Gamma, eps_inf=1):
        """
        Parameters
        ----------
        A  : float (in eV)
             Oscillator amplitude.
        E0 : float (in eV)
             Resonance energy.
        Gamma  : float (in eV)
             Oscillator width (damping constant).
        """
        Dispersion.__init__(self, eps_inf)
        self.A = A
        self.E0 = E0
        self.Gamma = Gamma

    def eps_1(self, E):
        return self.eps_inf + self.A * self.E0 * (self.E0 ** 2 - E ** 2) / \
            (self.E0 ** 2 - E ** 2) ** 2 + self.Gamma ** 2 * E ** 2

    def eps_2(self, E):
        return self.A * self.E0 * self.Gamma * E / \
            (self.E0 ** 2 - E ** 2) ** 2 + self.Gamma ** 2 * E ** 2


class TaucLorentz(Lorentz):

    def __init__(self, A, E0, Gamma, E_g, E_t, E_u, eps_inf=1):
        """
        Tauc assumed parabolic bands along with a constant momentum matrix
        element.
        
        Parameters
        ----------
        A : float
            Lorentz oscillator amplitude (oscillator strength) in eV.
        E0 : float
            Lorentz oscillator resonance (peak transition) energy in eV.
        Gamma : float
            Lorentz oscillator width (broadening) in eV.    
        E_g : float
            Optical band gap energy in eV.
        E_t : float
            The demarcation energy in eV between the Urbach tail transitions
            (E < E_t) and the band-to-band transitions (E > E_t). E_t should be
            greater than E_g. In case E_g = E_t there is no Urbach "tail".
        E_u : float
            The Urbach energy in eV.
        eps_inf : float (optional, default is 1.0)
            The limit of the real part of the dielectric constant.
        """
        Lorentz.__init__(self, A, E0, Gamma, eps_inf)
        self.E_g = E_g
        self.E_t = E_t
        self.E_u = E_u

    def _I_TL(self, E, a0, a1, a2, a3, c0, d0):
        """
        Auxiliary function. Eq. (5)
        """
        com_mul = 2 / math.pi * self.A * self.E0 * self.Gamma
        term_a3 = a3 * (self._zeta2 * self._I_1T - 0.25 * math.log((self._L_D(self.E_t))))
        term_a2 = a2 * (self._I_0AT + self._I_0BT)
        term_a1 = a1 * self._I_1T
        term_a0 = a0 * (self._I_0AT - self._I_0BT) / self.E0 ** 2
        term_c0 = -c0 * math.log(abs(E - self.E_t))
        term_d0 = -d0 * math.log(E + self.E_t)
        return com_mul * (term_a3 + term_a2 + term_a1 + term_a0 + term_c0 + term_d0)

    @property
    def _I_1T(self):
        """Auxiliary quantity. Eq. (6)"""
        com_mul = 1 / (2 * self._chi * self.Gamma)
        arg = 2 * (self.E_t ** 2 - self._zeta2) / (self._chi * self.Gamma)
        return com_mul * (math.pi - 2 * math.atan(arg))

    @property
    def _I_0AT(self):
        """Auxiliary quantity. Eq. (7)"""
        com_mul = 1 / (2 * self.Gamma)
        arg_1 = (self._chi + 2 * self.E_t) / self.Gamma
        arg_2 = (self._chi - 2 * self.E_t) / self.Gamma
        return com_mul * (math.pi - math.atan(arg_1) + math.atan(arg_2))

    @property
    def _I_0BT(self):
        """Auxiliary quantity. Eq. (8)"""
        pre_ln = 1 / (4 * self._chi)
        arg_num = self.E_t ** 2 + self.E0 ** 2 + self._chi * self.E_t
        arg_den = self.E_t ** 2 + self.E0 ** 2 - self._chi * self.E_t
        return pre_ln * math.log(arg_num / arg_den)

    @property
    def _zeta2(self):
        """Auxiliary quantity. Eq. (9) squared
           Note: always real
        """
        return self.E0 ** 2 - self.Gamma ** 2 / 2

    @property
    def _chi(self):
        """Auxiliary quantity. Eq. (10)
           Note: depending on the E0 and Gamma, could be complex
        """
        return (4 * self.E0 ** 2 - self.Gamma ** 2) ** 0.5

    def _c_0T(self, E):
        """
        Auxiliary function. Eq. (11)
        """
        return E * self._G_T(E) / (2 * self._L_D(E))

    def _d_0T(self, E):
        """
        Auxiliary function. Eq. (12)
        """
        num = -(E + self.E_g) ** 2
        den = 2 * E * self._L_D(E)
        return num / den

    def _a_3T(self, E):
        """
        Auxiliary function. Eq. (13)
        """
        return -(self._c_0T(E) + self._d_0T(E))

    def _a_2T(self, E):
        """
        Auxiliary function. Eq. (14)
        """
        return -E * (self._c_0T(E) - self._d_0T(E))

    def _a_1T(self, E):
        """
        Auxiliary function. Eq. (15)
        """
        return -(E ** 2 - 2 * self._zeta2) * (self._c_0T(E) + self._d_0T(E))

    def _a_0T(self, E):
        """
        Auxiliary function. Eq. (16)
        """
        return 1 - E * (E ** 2 - 2 * self._zeta2) * (self._c_0T(E) - self._d_0T(E))

    def _L_D(self, E):
        """
        Auxiliary function (the denominator of the Lorentz oscillator
        function). Eq. (17)
        """
        return (E ** 2 - self.E0 ** 2) ** 2 + self.Gamma ** 2 * E ** 2

    def _G_T(self, E):
        """
        Empirical variable band edge function, that forces eps_2 to assume a
        desired form for energies just above the E_g. For higher energies the
        Lorentz form holds.
        """
        num = (E - self.E_g) ** 2
        den = E ** 2
        return num / den

    def _L(self, E):
        """
        Lorentz oscillator function.
        """
        num = self.A * self.E0 * self.Gamma * E
        return num / self._L_D(E)

    @property
    def _E_1_T(self):
        """This constant governs the continuity between Urbach and Tauc
        parts.
        """
        return self.E_t * self._L(self.E_t) * self._G_T(self.E_t)

    def _I_U(self, E, E1):
        """
        Auxiliary function. Eq. (32)
        ToDo : there is the possibility to improve convergence because exp * Ei cancels divergences of both functions.
        """
        com_mul = E1 / (math.pi * E)
        term_1 = math.exp((E - self.E_t) / self.E_u)
        term_2 = Ei((self.E_t - E) / self.E_u)
        term_3 = Ei(-E / self.E_u)
        term_4 = math.exp(-(E + self.E_t) / self.E_u)
        term_5 = Ei((self.E_t + E) / self.E_u)
        term_6 = Ei(E / self.E_u)
        return com_mul * (term_1 * (term_2 - term_3) - term_4 * (term_5 - term_6))

    def eps_1(self, E):
        """
        Real part of the dielectric function. Eq. (2)
        """
        return self.eps_inf + \
            self._I_U(E, self._E_1_T) + \
            self._I_TL(E, self._a_0T(E), self._a_1T(E), self._a_2T(E),
                       self._a_3T(E), self._c_0T(E), self._d_0T(E))

    def eps_2(self, E):
        """
        Imaginary part of the dielectric function.
        """
        if 0 < E <= self.E_t:  # Urbach absorption
            pre_exp = self._E_1_T / E
            return pre_exp * math.exp((E - self.E_t) / self.E_u)
        elif E > self.E_t:  # band-to-band absorption
            return self._L(E) * self._G_T(E)
        else:  # negative energies
            raise ValueError(
                'All the negative energies are in the parallel universe, mate.')


class CodyLorentz:

    def __init__(self, A, E_0, Gamma, E_g, E_p, E_t, E_u, eps_inf=1):
        """
        Cody proposed applying a constant dipole matrix element rather than a
        constant momentum matrix element.
        
        Parameters
        ----------
        A : float
            Lorentz oscillator amplitude (oscillator strength) in eV.
        E_0 : float
            Lorentz oscillator resonance (peak transition) energy in eV.
        Gamma : float
            Lorentz oscillator width (broadening) in eV.
        E_g : float
            Optical band gap energy in eV.
        E_p : float
            Transition energy in eV, that separates the absorption onset behaviour
            (E < E_p + E_g) from the Lorentz oscillator behaviour
            (E > E_p + E_g).
        E_t : float
            The demarcation energy in eV between the Urbach tail transitions
            (E < E_t) and the band-to-band transitions (E > E_t).
        E_u : float
            The Urbach energy in eV.
        eps_inf : float (optional, default is 1.0)
            The limit of the real part of the dielectric constant.
        """
        self.A = A
        self.E_0 = E_0
        self.Gamma = Gamma
        self.E_g = E_g
        self.E_p = E_p
        self.E_t = E_t
        self.E_u = E_u
        self.eps_inf = eps_inf

    @property
    def _zeta2(self):
        """Auxiliary quantity. Eq. 9"""
        return self.E_0 ** 2 - self.Gamma ** 2 / 2

    @property
    def _chi(self):
        """Auxiliary quantity. Eq. 10"""
        return math.sqrt(4 * self.E_0 ** 2 - self.Gamma ** 2)

    @property
    def _I_1T(self):
        """Auxiliary quantity. Eq. 6"""
        com_mul = 1 / (2 * self._chi * self.Gamma)
        arg = 2 * (self.E_t ** 2 - self._zeta2) / (self._chi * self.Gamma)
        return com_mul * (math.pi - 2 * math.atan(arg))

    @property
    def _I_0AT(self):
        """Auxiliary quantity. Eq. 7"""
        com_mul = 1 / (2 * self.Gamma)
        arg_1 = (self._chi + 2 * self.E_t) / self.Gamma
        arg_2 = (self._chi - 2 * self.E_t) / self.Gamma
        return com_mul * (math.pi - math.atan(arg_1) + math.atan(arg_2))

    @property
    def _I_0BT(self):
        """Auxiliary quantity. Eq. 8"""
        pre_ln = 1 / (4 * self._chi)
        arg_num = self.E_t ** 2 + self.E_0 ** 2 + self._chi * self.E_t
        arg_den = self.E_t ** 2 + self.E_0 ** 2 - self._chi * self.E_t
        return pre_ln * math.log(arg_num / arg_den)

    @property
    def _I_0C(self):
        """Auxiliary quantity. Eq. 20"""
        arg = (self.E_t - self.E_g) / self.E_p
        return 1 / self.E_p * (math.pi / 2 - math.atan(arg))

    @property
    def _F(self):
        """Auxiliary quantity."""
        return math.sqrt(self.E_p ** 2 + self.E_g ** 2)

    @property
    def _K2(self):
        """Auxiliary quantity, Eq. (30)."""
        return 2 * self._F ** 2 + 2 * self._zeta2 - 4 * self.E_g ** 2

    @property
    def _Y4(self):
        """Auxiliary quantity."""
        term_1 = self.E_0 ** 4
        term_2 = self._F ** 2 * (self._K2 - self._F ** 2)
        term_3 = -4 * self.E_g ** 2 * self._K2
        return term_1 + term_2 + term_3

    def _G_C(self, E):
        """
        Empirical variable band edge function, that forces eps_2 to assume a
        desired form for energies just above the E_g. For higher energies the
        Lorentz form holds.
        """
        num = (E - self.E_g) ** 2
        den = num + self.E_p ** 2
        return num / den

    def _L_D(self, E):
        """
        Auxiliary function (the denominator of the Lorentz oscillator function).
        """
        return (E ** 2 - self.E_0 ** 2) ** 2 + self.Gamma ** 2 * E ** 2

    def _L(self, E):
        """
        Lorentz oscillator function.
        """
        num = self.A * self.E_0 * self.Gamma * E
        return num / self._L_D(E)

    def eps_2(self, E):
        """
        Imaginary part of the dielectric function.
        """
        if 0 < E <= self.E_t:  # Urbach absorption
            pre_exp = self.E_t * self._L(self.E_t) * self._G_C(self.E_t) / E
            return pre_exp * math.exp((E - self.E_t) / self.E_u)
        elif E > self.E_t:  # band-to-band absorption
            return self._L(E) * self._G_C(E)
        else:  # negative energies
            raise ValueError(
                'All the negative energies are in the parallel universe, mate.')

    def _I_TL(self, E, a0, a1, a2, a3, c0, d0):
        """
        Auxiliary function.
        """
        com_mul = 2 / math.pi * self.A * self.E_0 * self.Gamma
        term_a3 = a3 * (self._zeta2 * self._I_1T -
                        math.log(math.sqrt(math.sqrt(self._L_D(self.E_t)))))
        term_a2 = a2 * (self._I_0AT + self._I_0BT)
        term_a1 = a1 * self._I_1T
        term_a0 = a0 * (self._I_0AT - self._I_0BT) / self.E_0 ** 2
        term_c0 = -c0 * math.log(abs(E - self.E_t))
        term_d0 = -d0 * math.log(E + self.E_t)
        return com_mul * (
                term_a3 + term_a2 + term_a1 + term_a0 + term_c0 + term_d0)

    def _a_0C(self, E):
        """
        Auxiliary function.
        """
        term_1 = (self._K2 - self._F ** 2) * self._b_0C(E)
        term_2 = 2 * self.E_g * self._K2 * self._b_1C(E)
        term_3 = -E * (E ** 2 - 2 * self._zeta2) * (self._c_0C(E) -
                                                        self._d_0C(E))
        return 1 + term_1 + term_2 + term_3

    def _a_1C(self, E):
        """
        Auxiliary function.
        """
        term_1 = -2 * self.E_g * self._b_0C(E)
        term_2 = (self._K2 - self._F ** 2) * self._b_1C(E)
        term_3 = -(E ** 2 - 2 * self._zeta2) * (self._c_0C(E) + self._d_0C(E))
        return term_1 + term_2 + term_3

    def _a_2C(self, E):
        """
        Auxiliary function.
        """
        term_1 = -self._b_0C(E)
        term_2 = -2 * self.E_g * self._b_1C(E)
        term_3 = -E * (self._c_0C(E) - self._d_0C(E))
        return term_1 + term_2 + term_3

    def _a_3C(self, E):
        """
        Auxiliary function.
        """
        return -self._b_1C(E) - self._c_0C(E) - self._d_0C(E)

    def _b_0C(self, E):
        """
        Auxiliary function.
        """
        num_com_mul = self._Y4 * self._F ** 2
        num_1 = (self._c_0C(E) - self._d_0C(E)) / E
        num_2 = 2 * self.E_g * self._K2 / self._Y4 * (self._c_0C(E) +
                                                              self._d_0C(E))
        num = num_com_mul * (self._L_D(E) * (num_1 + num_2) - 1)
        den_1 = (self._K2 - self._F ** 2) * self._F ** 2 * self._Y4
        den_2 = self.E_0 ** 4 * self._Y4
        den_3 = 4 * self.E_g ** 2 * self._F ** 2 * self._K2 ** 2
        den = den_1 + den_2 + den_3
        return num / den

    def _b_1C(self, E):
        """
        Auxiliary function.
        """
        com_mul = 1 / self._Y4
        term_1 = 2 * self.E_g * self._K2 * self._b_0C(E)
        term_2 = -self._L_D(E) * (self._c_0C(E) + self._d_0C(E))
        return com_mul * (term_1 + term_2)

    def _c_0C(self, E):
        """
        Auxiliary function.
        """
        return E * self._G_C(E) / (2 * self._L_D(E))

    def _d_0C(self, E):
        """
        Auxiliary function.
        """
        num = -E * (E + self.E_g) ** 2
        den = 2 * self._L_D(E) * ((E + self.E_g) ** 2 + self.E_p ** 2)
        return num / den

    def _I_CL(self, E, a0, a1, a2, a3, b0, b1, c0, d0):
        """
        Auxiliary function.
        """
        term_1 = self._I_TL(E, a0, a1, a2, a3, c0, d0)
        pre_mul = 2 * self.A * self.E_0 * self.Gamma / math.pi
        term_2 = self.E_g * self._I_0C
        term_3 = math.log(math.sqrt((self.E_t - self.E_g) ** 2 + self.E_p ** 2))
        term_4 = b0 * self._I_0C
        return term_1 + pre_mul * (b1 * (term_2 - term_3) + term_4)

    def _I_U(self, E):
        """
        Auxiliary function.
        """
        com_mul = self.E_t * self._L(self.E_t) * self._G_C(self.E_t) / (math.pi * E)
        term_1 = math.exp((E - self.E_t) / self.E_u)
        term_2 = Ei((self.E_t - E) / self.E_u)
        term_3 = Ei(-E / self.E_u)
        term_4 = math.exp(-(E + self.E_t) / self.E_u)
        term_5 = Ei((self.E_t + E) / self.E_u)
        term_6 = Ei(E / self.E_u)
        return com_mul * (term_1 * (term_2 - term_3) - term_4 * (term_5 - term_6))

    def eps_1(self, E):
        """
        Real part of the dielectric function.
        """
        return self.eps_inf + \
            self._I_U(E) + \
            self._I_CL(E, self._a_0C(E), self._a_1C(E), self._a_2C(E),
                       self._a_3C(E), self._b_0C(E), self._b_1C(E),
                       self._c_0C(E), self._d_0C(E))

    def n(self, E):
        """
        Index of refraction.
        """
        e1 = self.eps_1(E)
        e2 = self.eps_2(E)
        return math.sqrt((math.hypot(e1, e2) + e1) / 2)

    def k(self, E):
        """
        Extinction coefficient.
        """
        e1 = self.eps_1(E)
        e2 = self.eps_2(E)
        return math.sqrt((math.hypot(e1, e2) - e1) / 2)

    def R(self, E):
        """
        Reflectivity.
        """
        n = self.n(E)
        k = self.k(E)
        num = (n - 1) ** 2 + k ** 2
        den = (n + 1) ** 2 + k ** 2
        return num / den

    def a(self, E):
        """
        Absorption coefficient.
        """
        k = self.k(E)
        wavelength = 1.239841973862093e-06 / E
        return 4 * math.pi * k / wavelength


if __name__ == '__main__':
    tl = Foldyna(128.6, 2.52, 3.93, 0.67, 0.7)  # from (Němec, 2009)
    Energy = 1239.841973862093 / 1030
    print(f'{Energy} eV')
    print('eps_1 = {:.2f}'.format(tl.eps_1(Energy)))
    print('eps_2 = {:.2f}'.format(tl.eps_2(Energy)))
    quit()
    print('n = {:.4g}'.format(tl.n(Energy)))
    print('k = {:.4g}'.format(tl.k(Energy)))
    print('R = {:.3g}%'.format(tl.R(Energy) * 100))
    print('a = {:.3g}'.format(tl.a(Energy)))

    import numpy as np

    w = np.linspace(500, 2500, 501) * 1e-9  # nm
    y = []
    for i in range(len(w)):
        Energy = 1.239841973862093e-06 / w[i]
        y.append([w[i] * 1e9, tl.n(Energy), tl.k(Energy)])

    y = np.array(y)
    np.savetxt('GST225 TL (w, n, k).txt', y)

    import matplotlib.pyplot as plt

    x = 1239.841973862093e-9 / w
    plt.plot(y[:, 0], y[:, 2])
    plt.yscale('log')
    plt.show()
