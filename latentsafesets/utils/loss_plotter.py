import os
import logging

log = logging.getLogger("")


class LossPlotter:
    def __init__(self, logdir):
        self.logdir = logdir
        os.makedirs(logdir, exist_ok=True)
        self.data = {}
        self.running_avgs = {}

    def add_data(self, data):
        for key in data:
            if key in self.running_avgs:
                ra = self.running_avgs[key]
                ra = ra * 0.9 + data[key] * 0.1
                self.running_avgs[key] = ra
                self.data[key].append(ra)
            else:
                ra = data[key]
                self.running_avgs[key] = ra
                self.data[key] = [ra]

    def plot(self):
        import latentsafesets.utils.plot_utils as pu
        
        for key in self.data:
            losses = self.data[key]
            fname = os.path.join(self.logdir, '%s_loss.pdf' % key)
            pu.loss_plot(losses, fname, title='%s loss' % key)

    def print(self, i=None, other_data=None):
        if other_data is None:
            other_data = {}
        log.info('--------------------')
        if i is not None:
            log.info("%s: %d" % ("iter".ljust(25), i))
        for key in self.data:
            log.info("%s: %.5f" % (key.ljust(25), self.running_avgs[key]))
        for key in other_data:
            log.info("%s: %s" % (key.ljust(25), str(other_data[key])))
