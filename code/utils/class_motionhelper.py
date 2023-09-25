import numpy as np
from datetime import datetime


class Timer(object):
    def __init__(self, HZ, MAX_SEC, VERBOSE=True):
        self.time_start     = datetime.now()
        self.sec_next       = 0.0
        self.HZ             = HZ
        if self.HZ == 0:
            self.sec_period  = 0
        else:
            self.sec_period  = 1.0 / self.HZ
        self.max_sec        = MAX_SEC
        self.sec_elps       = 0.0
        self.sec_elps_prev  = 0.0
        self.sec_elps_diff  = 0.0
        self.tick           = 0.
        self.force_finish   = False
        self.DELAYED_FLAG   = False
        self.VERBOSE        = VERBOSE
        print ("TIMER WITH [%d]HZ INITIALIZED. MAX_SEC IS [%.1fsec]."
            % (self.HZ, self.max_sec))

    def start(self):
        self.time_start     = datetime.now()
        self.sec_next       = 0.0
        self.sec_elps       = 0.0
        self.sec_elps_prev  = 0.0
        self.sec_elps_diff  = 0.0
        self.tick           = 0.

    def finish(self):
        self.force_finish = True

    def is_finished(self):
        self.time_diff = datetime.now() - self.time_start
        self.sec_elps  = self.time_diff.total_seconds()
        if self.force_finish:
            return True
        if self.sec_elps > self.max_sec:
            return True
        else:
            return False

    def is_notfinished(self):
        self.time_diff = datetime.now() - self.time_start
        self.sec_elps  = self.time_diff.total_seconds()
        if self.force_finish:
            return False
        if self.sec_elps > self.max_sec:
            return False
        else:
            return True

    def do_run(self):
        self.time_diff = datetime.now() - self.time_start
        self.sec_elps  = self.time_diff.total_seconds()
        if self.sec_elps > self.sec_next:
            self.sec_next = self.sec_next + self.sec_period
            self.tick     = self.tick + 1
            """ COMPUTE THE TIME DIFFERENCE & UPDATE PREVIOUS ELAPSED TIME """
            self.sec_elps_diff = self.sec_elps - self.sec_elps_prev
            self.sec_elps_prev = self.sec_elps
            """ CHECK DELAYED """
            if (self.sec_elps_diff > self.sec_period*1.5) & (self.HZ != 0):
                if self.VERBOSE:
                    print ("sec_elps_diff:[%.1fms]" % (self.sec_elps_diff*1000.0))
                    print ("[%d][%.1fs] DELAYED! T:[%.1fms] BUT IT TOOK [%.1fms]"
                        % (self.tick, self.sec_elps, self.sec_period*1000.0, self.sec_elps_diff*1000.0))
                self.DELAYED_FLAG = True
            return True
        else:
            return False