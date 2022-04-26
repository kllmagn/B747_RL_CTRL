import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class LivePlotter:
    def __init__(self, data_callback=None, data_limit=500):
        plt.ion()
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.xs = []
        self.ys = []
        self.data_limit = data_limit
        self.data_callback = data_callback

    def set_callback(self, data_callback):
        self.data_callback = data_callback

    def animate(self, i):
        if not self.data_callback:
            raise ValueError("Data callback was not set!")
        self.data_callback(self.xs, self.ys)

        self.xs = self.xs[-100:]
        self.ys = self.ys[-100:]

        self.ax.clear()
        self.ax.plot(self.xs, self.ys)

        '''
        plt.xticks(rotation=45, ha='right')
        plt.subplots_adjust(bottom=0.30)
        plt.title('TMP102 Temperature over Time')
        plt.ylabel('Temperature (deg C)')
        '''

    def show(self, update_interval=1000):
        self.ani = animation.FuncAnimation(self.fig, self.animate, interval=update_interval)
        #plt.show()

    def clear(self):
        self.xs.clear()
        self.ys.clear()
        #plt.close()