import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from datetime import datetime, timedelta
from scipy import fftpack, integrate, signal
plt.style.use('seaborn')
mpl.rcParams['axes.labelsize'] = 14
mpl.rcParams['figure.titlesize'] = 18
mpl.rcParams['xtick.labelsize'] = 12

def get_data():
    """
    Read CSV
    """
    # Skip rows until start of data except heading
    # ipad_rows = [0, 1] + [i for i in range(3, 408)]
    ipad = pd.read_csv("data/ipad3_formatted.csv")#, skiprows=ipad_rows)
    iphone = pd.read_csv("data/iphone3.csv")
    # print(ipad.columns.values)
    # ipad.to_csv("ipad.csv", encoding="utf-8")

    # Set condition start time
    start = datetime(2020, 4, 26, 9, 14, 16)

    # Grab ios imu data
    ipad_time_raw = ipad['loggingTime(txt) ']
    # Convert time to DateTime object with 4 hours added to get UTC time
    ipad_time_uf = np.array([datetime.strptime(d[:-6], "%Y-%m-%d %H:%M:%S.%f") for d in ipad_time_raw])
    # Numpy array of seconds fro condition start from IOS data
    ipad_time_uf -= start
    ipad_time = np.array([t.seconds + t.microseconds/1e6 for t in ipad_time_uf])

    acc_x = ipad[' accelerometerAccelerationX(G) ']
    acc_y = ipad[' accelerometerAccelerationY(G) ']
    acc_z = ipad[' accelerometerAccelerationZ(G) ']
    rot_rate_x = ipad[' motionRotationRateX(rad/s) ']
    rot_rate_y = ipad[' motionRotationRateY(rad/s) ']
    rot_rate_z = ipad[' motionRotationRateZ(rad/s) ']
    yaw = ipad[' motionYaw(rad) ']
    roll = ipad[' motionRoll(rad) ']
    pitch = ipad[' motionPitch(rad) ']

    # Grab ios imu data
    iphone_time_raw = iphone['loggingTime(txt)']
    # Convert time to DateTime object with 4 hours added to get UTC time
    iphone_time_uf = np.array([datetime.strptime(d[:-6], "%Y-%m-%d %H:%M:%S.%f")+timedelta(microseconds=0) for d in iphone_time_raw])
    iphone_time_uf -= start
    # Numpy array of seconds fro condition start from IOS data
    iphone_time = np.array([t.seconds + t.microseconds/1e6 for t in iphone_time_uf])

    acc_x2 = iphone['accelerometerAccelerationX(G)']
    acc_y2 = iphone['accelerometerAccelerationY(G)']
    acc_z2 = iphone['accelerometerAccelerationZ(G)']
    gyro_x2 = iphone['motionRotationRateX(rad/s)']
    gyro_y2 = iphone['motionRotationRateY(rad/s)']
    gyro_z2 = iphone['motionRotationRateZ(rad/s)']
    yaw2 = iphone['motionYaw(rad)']
    roll2 = iphone['motionRoll(rad)']
    pitch2 = iphone['motionPitch(rad)']
    print(ipad_time[:5])
    print(iphone_time[:5])
    t_range1 = np.where(np.logical_and(ipad_time > 10, ipad_time < 310))[0]
    t_range2 = np.where(np.logical_and(iphone_time > 10, iphone_time < 310))[0]
    print("iPad freq:", len(t_range1)/300, "iPhone freq:", len(t_range2)/300)

    # Check taps with ipad and iphone for time Synchronization
    plt.figure()
    acc_tot = np.linalg.norm([acc_x[:300], acc_y[:300], acc_z[:300]+1], axis=0)
    acc_tot2 = np.linalg.norm([acc_x2[:300], acc_y2[:300], acc_z2[:300]+1], axis=0)/2
    plt.plot(ipad_time[:300], acc_tot, alpha=0.5, label=r"$|a_{ipad}|$")
    plt.plot(iphone_time[:300], acc_tot2, alpha=0.5, color="red", label=r"$|a_{iphone}|$")
    plt.ylabel("Total acceleration")
    plt.xlabel("Time from Test Start [seconds]")
    plt.title("Time Synchronization")
    plt.legend()
    plt.tight_layout()
    plt.show()

    def plot_roll(t1, t2, title, ylim1=-50, ylim2=50, ylim3=-300, ylim4=300):
        # Get time indices for condition
        tg1, tg2, ti1, ti2 = get_indices(t1, t2, ipad_time, iphone_time)
        # Integrate gyro angular rate for absolut angle
        y = integrate.cumtrapz(gyro_y2[ti1:ti2:1], iphone_time[ti1:ti2:1], initial=0)
        # FFT
        # freq_x = fftpack.fft(signal.wiener(gyro_y2[ti1:ti2]*180/np.pi))
        freq_x = fftpack.fft(np.array(y*180/np.pi))
        fftfreq_x = fftpack.fftfreq(len(freq_x), 1/33.3)
        i = fftfreq_x > 0
        # high_freq_fft_x = freq_x.copy()
        # high_freq_fft_x[np.abs(fftfreq_x) > 10] = 0
        # filtered_sig_x = fftpack.ifft(high_freq_fft_x)
        # freq_y = fftpack.fft(signal.wiener(roll[tg1:tg2]*180/np.pi))
        freq_y = fftpack.fft(np.array(roll[tg1:tg2]*180/np.pi))
        fftfreq_y = fftpack.fftfreq(len(freq_y), 1/29.52)
        j = fftfreq_y > 0
        # high_freq_fft = freq_y.copy()
        # high_freq_fft[np.abs(fftfreq_y) > 2] = 0
        # filtered_sig = fftpack.ifft(high_freq_fft)

        # Periodogram & PSD
        # fx, Pxx_den = signal.periodogram(y, 33.3)
        # fy, Pyy_den = signal.periodogram(roll[tg1:tg2]*180/np.pi, 29.52)
        #
        # plt.figure()
        # plt.scatter(fx, Pxx_den)
        # plt.scatter(fy, Pyy_den)
        # plt.xlabel('frequency [Hz]')
        # plt.ylabel('PSD [deg^2/Hz]')

        # Plot time data - filtered data commented out
        f, (ax1, ax2) = plt.subplots(2, 1, figsize=(7,7))
        ax1.plot(ipad_time[tg1:tg2], roll[tg1:tg2]*180/np.pi, "green", label="Roll Angle", alpha=0.8)
        ax1.plot(iphone_time[ti1:ti2], y*180/np.pi, label="Yoke Roll Angle", color="blue", alpha=0.8)
        # ax1.plot(ipad_time[tg1:tg2], filtered_sig, label="Filtered Roll Angle", color="red", alpha=0.8)
        # ax1.plot(ipad_time[tg1:tg2], signal.wiener(roll[tg1:tg2]*180/np.pi), label="Wiener")
        ax1.set_ylabel("Angle [deg]")
        ax1.legend()
        ax1.set_title(title)
        ax2.plot(iphone_time[ti1:ti2], gyro_y2[ti1:ti2]*180/np.pi, label="Yoke Roll Rate [rad/s]", color="maroon")
        # ax2.plot(iphone_time[ti1:ti2], filtered_sig_x, label="Filtered Yoke Roll Rate [rad/s]")
        # ax2.plot(iphone_time[ti1:ti2], signal.wiener(gyro_y2[ti1:ti2]*180/np.pi), label="Wiener")
        ax2.legend()
        ax2.set_ylabel("Angular Rate [deg]")
        ax2.set_xlabel("Time from Test Start [seconds]")
        plt.tight_layout()
        filename = title.replace(" ", "_").lower()
        # plt.savefig("plots/" + filename + ".png", dpi=600)

        # Plot bode plot
        f, (ax1, ax2) = plt.subplots(2, 1, figsize=(7,7))
        idx = int(len(freq_y)/5)
        ax1.scatter(fftfreq_x[i], np.abs(freq_x[i]), label=r"$|X(j\omega)|$", alpha=0.8)
        ax1.scatter(fftfreq_y[j], np.abs(freq_y[j]), label=r"$|Y(j\omega)|$", alpha=0.8)
        # ax1.plot(fftfreq_x[:idx], np.abs(freq_y[:idx])/np.abs(freq_x[:idx]), label=r"|$H(j\omega)|$")
        ax1.set_title("Frequency Response")
        # ax1.set_xscale('log')
        # ax1.set_xlabel("Frequency [Hz]")
        ax1.set_ylabel("magnitude$")
        # ax1.legend()

        ax2.scatter(fftfreq_x[i], np.angle(freq_x[i], deg=True), label=r"$\phi X(j\omega)$", alpha=0.8)
        ax2.scatter(fftfreq_y[j], np.angle(freq_y[j], deg=True), label=r"$\phi Y(j\omega)$", alpha=0.8)
        # ax2.plot(fftfreq_x[:idx], np.angle(freq_y[:idx], deg=True)-np.angle(freq_x[:idx], deg=True), label=r"$\phi Y(j\omega)$")
        ax2.set_xlabel("Frequency [Hz]")
        ax2.set_ylabel("Phase [deg]")
        # ax2.set_xscale('log')
        plt.tight_layout()
        # plt.savefig("plots/" + filename + "_freq.png", dpi=600)
        plt.show()

    def plot_sequence(ts, title):
        f, axes = plt.subplots(len(ts), 1, figsize=(7,7))
        for i, (t1, t2) in enumerate(ts):
            tg1, tg2, ti1, ti2 = get_indices(t1, t2, ipad_time, iphone_time)
            y = integrate.cumtrapz(gyro_y2[ti1:ti2:1], iphone_time[ti1:ti2:1], initial=0)
            axes[i].plot(iphone_time[ti1:ti2], y*180/np.pi, label="Yoke Roll Angle", color="blue", alpha=0.8)
            axes[i].plot(ipad_time[tg1:tg2], roll[tg1:tg2]*180/np.pi, "green", label="Roll Angle", alpha=0.8)
        axes[0].legend()
        axes[-1].set_xlabel("Time from Test Start [sec]")
        f.suptitle(title)
        filename = title.replace(" ", "_").lower()
        plt.savefig("plots/time/" + filename, dpi=600)
        plt.show()

    def freq_analysis(ts, title):
        """ Assumes equal time intervals in ts
        """
        fx = [[] for _ in range(len(ts))]
        fy = [[] for _ in range(len(ts))]
        Pxx = [[] for _ in range(len(ts))]
        Pyy = [[] for _ in range(len(ts))]
        tdelta1 = None
        tdelta2 = None
        for i, (t1, t2) in enumerate(ts):
            tg1, tg2, ti1, ti2 = get_indices(t1, t2, ipad_time, iphone_time)
            if tdelta1 == None:
                tdelta1 = tg2 - tg1
                tdelta2 = ti2 - ti1
            else:
                tg2 = tg1 + tdelta1
                ti2 = ti1 + tdelta2

            y = integrate.cumtrapz(gyro_y2[ti1:ti2:1], iphone_time[ti1:ti2:1], initial=0)
            fx[i], Pxx[i] = signal.periodogram(y*180/np.pi, 33.3)
            fy[i], Pyy[i] = signal.periodogram(roll[tg1:tg2]*180/np.pi, 29.52)
        plt.figure()
        idx = int(len(fx[0])/5)
        # plt.plot(fx[0][:idx], np.mean(Pxx, axis=0)[:idx]/np.mean(Pyy, axis=0)[:idx])
        plt.scatter(fx[0], np.mean(Pxx, axis=0))
        plt.scatter(fy[0], np.mean(Pyy, axis=0))
        plt.xlabel("Frequency [Hz]")
        # plt.ylabel(r"$|H(e^{j\Omega})|^2$")
        plt.ylabel(r"PSD $|S(e^{j\Omega})|$")
        plt.title(title)
        plt.show()

    # (181.8, 183.5), (185, 186.5)
    # plot_roll(181.8, 183.5, "Roll Doublet Fast 1")
    # plot_roll(185, 186.5, "Roll Doublet Fast 2")
    # plot_roll(188, 192, "Roll Doublet Fast 3")
    # # # (188, 191.5), (191, 195)
    # plot_roll(188, 191.5, "Roll Doublet Med 1")
    # plot_roll(191, 195, "Roll Doublet Med 2")
    # # (196, 202), (204, 210)
    # plot_roll(196, 202, "Roll Doublet Slow 1")
    # plot_roll(204, 210, "Roll Doublet Slow 2")
    # # (212, 229), (231, 244)
    # plot_roll(214, 229, "Roll Frequency Sweep 1")
    # plot_roll(233.5, 244, "Roll Frequency Sweep 2")
    # (245, 257)
    # plot_roll(245, 257, "Step Response")

    plot_sequence([(182, 183.5), (185, 186.5)], "Roll Doublet Fast")
    plot_sequence([(188.5, 191.5), (191.5, 194.5)], "Roll Doublet Med")
    plot_sequence([(196, 202), (204, 210)], "Roll Doublet Slow")
    # plot_sequence([(214, 229), (233.5, 244)], "Roll Frequency Sweep")

    # freq_analysis([(182, 183.5), (185, 186.5)], "Roll Doublet Fast")
    # freq_analysis([(188.5, 191.5), (191.5, 194.5)], "Roll Doublet Med")
    # freq_analysis([(196, 202), (204, 210)], "Roll Doublet Slow")
    # plot_sequence([(214, 229), (233.5, 244)], "Roll Frequency Sweep")
    # plot_sequence([(245, 257)], "Roll Step Input")

def get_indices(t1, t2, d1, d2):
    """ Get indices in the array for start = t1 and end = t2 for d1 (iPad)
    and d2 (iPhone).
    Returns: i1_ipad, i2_ipad, i1_iphone, i2_iphone
    """
    t_range1 = np.where(np.logical_and(d1 > t1, d1 < t2))[0]
    t_range2 = np.where(np.logical_and(d2 > t1, d2 < t2))[0]
    return t_range1[0], t_range1[-1], t_range2[0], t_range2[-1]

if __name__ == "__main__":
    get_data()
