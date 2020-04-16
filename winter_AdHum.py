#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 14:56:35 2020

@author: cghiaus
"""
import numpy as np
import pandas as pd
import psychro as psy
import matplotlib.pyplot as plt

# global variables
UA = 935.83                 # bldg conductance
tIsp, wIsp = 18, 6.22e-3    # indoor conditions

# constants
c = 1e3                     # air specific heat J/kg K
l = 2496e3                  # latent heat J/kg

    
# *****************************************
# RECYCLED AIR
# *****************************************
def ModelRecAir(m, tS, mi, tO, phiO, alpha, beta):
    """
    Model:
        Heating and adiabatic humidification
        Recycled air
        CAV Constant Air Volume:
            mass flow rate calculated for design conditions
            maintained constant in all situations

    INPUTS:
        m     mass flow of supply dry air
        tS    supply air;                30°C for design conditions
        mi    infiltration massflow rate 2.18 kg/s for design conditions
        tO    outdoor temperature;       -1°C for design conditions
        phiO  outdoor relative humidity; 100% for design conditions
        alpha mixing ratio of outdoor air
        beta  by-pass factor of adiab. hum.

    System:
        MX1:    Mixing box
        HC1:    Heating Coil
        AH:     Adiabatic Humidifier
        MX2:    Mixing in humidifier model
        HC2:    Reheating coil
        TZ:     Thermal Zone
        BL:     Building
        Kw:     Controller - humidity
        Kt:     Controller - temperature
        o:      outdoor conditions
        0..5    unknown points (temperature, humidity ratio)

        <----|<-------------------------------------------|
             |                                            |
             |              |-------|                     |
        -o->MX1--0->HC1--1->|       MX2--3->HC2--4->TZ--5-|
                    |       |       |        |      ||    |
                    |       |->AH-2-|        |      BL    |
                    |                        |            |
                    |                        |<-----Kt----|<-t5
                    |<------------------------------Kw----|<-w5


    Returns
    -------
    x       vector 16 elem.:
            t0, w0, t1, w1, t2, w2, t3, w3, t4, w4, t5, w5,...
                QHC1, QHC2, QsTZ, QlTZ

    """
    Kt, Kw = 1e10, 1e10             # controller gain
    wO = psy.w(tO, phiO)            # hum. out

    # Model
    ts0, Del_ts = tS, 2             # initial guess saturation temp.

    A = np.zeros((16, 16))          # coefficents of unknowns
    b = np.zeros(16)                # vector of inputs
    while Del_ts > 0.01:
        # MX1
        A[0, 0], A[0, 10], b[0] = m*c, -(1 - alpha)*m*c, alpha*m*c*tO
        A[1, 1], A[1, 11], b[1] = m*l, -(1 - alpha)*m*l, alpha*m*l*wO
        # HC1
        A[2, 0], A[2, 2], A[2, 12], b[2] = m*c, -m*c, 1, 0
        A[3, 1], A[3, 3], b[3] = m*l, -m*l, 0
        # AH
        A[4, 2], A[4, 3], A[4, 4], A[4, 5], b[4] = c, l, -c, -l, 0
        A[5, 4], A[5, 5] = psy.wsp(ts0), -1
        b[5] = psy.wsp(ts0)*ts0 - psy.w(ts0, 1)
        # MX2
        A[6, 2], A[6, 4], A[6, 6], b[6] = beta*m*c, (1-beta)*m*c, -m*c, 0
        A[7, 3], A[7, 5], A[7, 7], b[7] = beta*m*l, (1-beta)*m*l, -m*l, 0
        # HC2
        A[8, 6], A[8, 8], A[8, 13], b[8] = m*c, -m*c, 1, 0
        A[9, 7], A[9, 9], b[9] = m*l, -m*l, 0
        # TZ
        A[10, 8], A[10, 10], A[10, 14], b[10] = m*c, -m*c, 1, 0
        A[11, 9], A[11, 11], A[11, 15], b[11] = m*l, -m*l, 1, 0
        # BL
        A[12, 10], A[12, 14], b[12] = (UA+mi*c), 1, (UA+mi*c)*tO
        A[13, 11], A[13, 15], b[13] = mi*l, 1, mi*l*wO
        # Kt & Kw
        A[14, 10], A[14, 12], b[14] = Kt, 1, Kt*tIsp
        A[15, 11], A[15, 13], b[15] = Kw, 1, Kw*wIsp

        x = np.linalg.solve(A, b)
        Del_ts = abs(ts0 - x[4])
        ts0 = x[4]
    return x


def RecAirCAV(tS=30, mi=2.18, tO=-1, phiO=1, alpha=1, beta=0.1):
    """
    Model:
        Heating and adiabatic humidification
        Recycled air
        CAV Constant Air Volume:
            mass flow rate calculated for design conditions
            maintained constant in all situations

    INPUTS:
        tS    supply air;                30°C for design conditions
        mi    infiltration massflow rate 2.18 kg/s for design conditions
        tO    outdoor temperature;       -1°C for design conditions
        phiO  outdoor relative humidity; 100% for design conditions
        alpha mixing ratio of outdoor air
        beta  by-pass factor of adiab. hum.

    System:
        MX1:    Mixing box
        HC1:    Heating Coil
        AH:     Adiabatic Humidifier
        MX2:    Mixing in humidifier model
        HC2:    Reheating coil
        TZ:     Thermal Zone
        BL:     Building
        Kw:     Controller - humidity
        Kt:     Controller - temperature
        o:      outdoor conditions
        0..5    unknown points (temperature, humidity ratio)

        <----|<-------------------------------------------|
             |                                            |
             |              |-------|                     |
        -o->MX1--0->HC1--1->|       MX2--3->HC2--4->TZ--5-|
                    |       |       |        |      ||    |
                    |       |->AH-2-|        |      BL    |
                    |                        |            |
                    |                        |<-----Kt----|<-t5
                    |<------------------------------Kw----|<-w5


    Returns
    -------
    None
    """
    plt.close('all')
    wO = psy.w(tO, phiO)            # hum. out

    # Mass flow rate for design conditions
    # Supplay air mass flow rate
    # QsZ = UA*(tO - tIsp) + mi*c*(tO - tIsp)
    # m = - QsZ/(c*(tS - tIsp)
    # where
    # tO, wO = -1, 3.5e-3           # outdoor
    # tS = 30                       # supply air
    # mi = 2.18                     # infiltration
    QsZ = UA*(-1 - tIsp) + 2.18*c*(-1 - tIsp)
    m = - QsZ/(c*(tS - tIsp))
    print(f'm = {m: 5.3f} kg/s constant for design conditions:')
    print(f'    [tSd = {tS: 3.1f} °C, mi = 2.18 kg/S, tO = -1°C, phi0 = 100%]')

    # Model
    x = ModelRecAir(m, tS, mi, tO, phiO, alpha, beta)

    t = np.append(tO, x[0:12:2])
    w = np.append(wO, x[1:12:2])

    A = np.array([[-1, 1,  0,  0,  0,  0, -1],
                 [0,  -1,  1,  0,  0,  0,  0],
                 [0,  0,  -1,  1,  0,  0,  0],
                 [0,  0,  -1, -1,  1,  0,  0],
                 [0,  0,   0,  0, -1,  1,  0],
                 [0,  0,   0,  0,  0, -1,  1]])

    psy.chartA(t, w, A)

    t = pd.Series(t)
    w = 1000*pd.Series(w)
    P = pd.concat([t, w], axis=1)       # points
    P.columns = ['t [°C]', 'w [g/kg]']

    output = P.to_string(formatters={
        't [°C]': '{:,.2f}'.format,
        'w [g/kg]': '{:,.2f}'.format
    })
    print()
    print(output)

    Q = pd.Series(x[12:], index=['QsHC1', 'QsHC2', 'QsTZ', 'QlTZ'])
    # Q.columns = ['kW']
    pd.options.display.float_format = '{:,.2f}'.format
    print()
    print(Q.to_frame().T/1000, 'kW')

    return None


def RecAirVAV(tSsp=30, mi=2.18, tO=-1, phiO=1, alpha=0.5, beta=0):
    """
    Created on Fri Apr 10 13:57:22 2020
    CAV Recirculated air Heating & Adiabatic Humidification
    mass flow rate calculated for design conditions
    maintained constant in all situations

    INPUTS:
    tS    supply air;                30°C for design conditions
    mi    infiltration massflow rate 2.18 kg/s for design conditions
    tO    outdoor temperature;       -1°C for design conditions
    phiO  outdoor relative humidity; 100% for design conditions
    alpha mixing ratio of outdoor air
    beta  by-pass factor of adiab. hum.

    System:
        MX1:    Mixing box
        HC1:    Heating Coil
        AH:     Adiabatic Humidifier
        MX2:    Mixing in humidifier model
        HC2:    Reheating coil
        TZ:     Thermal Zone
        BL:     Building
        Kw:     Controller - humidity
        Kt:     Controller - temperature
        o:      outdoor conditions
        0..5    unknown points (temperature, humidity ratio)

        <----|<-------------------------------------------------|
             |                                                  |
             |              |-------|                           |
        -o->MX1--0->HC1--1->|       MX2--3->HC2--F-----4->TZ--5-|
                    |       |->AH-2-|        |   |     |  ||    |
                    |                        |   |-Kt4-|  BL    |
                    |                        |                  |
                    |                        |<-----Kt----------|<-t5
                    |<------------------------------Kw----------|<-w5

    """
    from scipy.optimize import least_squares

    def Saturation(m):
        """
        Used in VAV to find the mass flow which solves tS = tSsp
        Parameters
        ----------
            m : mass flow rate of dry air

        Returns
        -------
            tS - tSsp: difference between supply temp. and its set point

        """
        x = ModelRecAir(m, tSsp, mi, tO, phiO, alpha, beta)
        tS = x[8]
        return (tS - tSsp)

    plt.close('all')
    wO = psy.w(tO, phiO)            # hum. out

    # Mass flow rate
    res = least_squares(Saturation, 5, bounds=(0, 10))
    if res.cost < 1e-10:
        m = float(res.x)
    else:
        print('RecAirVAV: No solution for m')

    print(f'm = {m: 5.3f} kg/s')
    x = ModelRecAir(m, tSsp, mi, tO, phiO, alpha, beta)

    # DtS, m = 2, 1                   # initial temp; diff; flow rate
    # while DtS > 0.01:
    #     m = m + 0.01
    #     # Model
    #     x = ModelRecAir(m, tSsp, mi, tO, phiO, alpha, beta)
    #     tS = x[8]
    #     DtS = -(tSsp - tS)

    t = np.append(tO, x[0:12:2])
    w = np.append(wO, x[1:12:2])

    A = np.array([[-1, 1,  0,  0,  0,  0, -1],
                 [0,  -1,  1,  0,  0,  0,  0],
                 [0,  0,  -1,  1,  0,  0,  0],
                 [0,  0,  -1, -1,  1,  0,  0],
                 [0,  0,   0,  0, -1,  1,  0],
                 [0,  0,   0,  0,  0, -1,  1]])

    psy.chartA(t, w, A)

    t = pd.Series(t)
    w = 1000*pd.Series(w)
    P = pd.concat([t, w], axis=1)       # points
    P.columns = ['t [°C]', 'w [g/kg]']

    output = P.to_string(formatters={
        't [°C]': '{:,.2f}'.format,
        'w [g/kg]': '{:,.2f}'.format
    })
    print()
    print(output)

    Q = pd.Series(x[12:], index=['QsHC1', 'QsHC2', 'QsTZ', 'QlTZ'])
    # Q.columns = ['kW']
    pd.options.display.float_format = '{:,.2f}'.format
    print()
    print(Q.to_frame().T/1000, 'kW')
  
    return None


# RecAirCAV()
# RecAirVAV()
# RecAirVAV(tSsp=30, mi=2.18, tO=-1, phiO=1, alpha=0.5, beta=0.50)
# tSsp=20, mi=2.18, tO=-1, phiO=1, alpha=1, beta=0
# RecAirVAV(20, 2.18, -1, 1, 1, 0.1)
# ModelRecAir(m, tSsp, mi, tO, phiO, alpha, beta)
