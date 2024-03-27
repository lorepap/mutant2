class Experiment:
    def __init__(self, name):
       
        self.name = name
        self.settings = self.experiments()[name]

    @staticmethod
    def experiments():
        return  {
            "Baseline": {"bw": 42, "rtt": 20, "bdp_mult": 1, "bw_factor": 1},
            "Low_Bandwidth": {"bw": 6, "rtt": 20, "bdp_mult": 1, "bw_factor": 1},
            "High_RTT": {"bw": 12, "rtt": 80, "bdp_mult": 1, "bw_factor": 1},
            "Large_Queue": {"bw": 12, "rtt": 20, "bdp_mult": 10, "bw_factor": 1},
            "Mixed_Conditions": {"bw": 42, "rtt": 30, "bdp_mult": 2, "bw_factor": 1},
            "Challenging_Network": {"bw": 6, "rtt": 100, "bdp_mult": 1, "bw_factor": 1},
            "Challenging_Network_2": {"bw": 12, "rtt": 30, "bdp_mult": 0.5, "bw_factor": 1},
            "wired6": {"bw": 6, "rtt": 20, "bdp_mult": 1, "bw_factor": 1},
            "wired12": {"bw": 12, "rtt": 20, "bdp_mult": 1, "bw_factor": 1},
            "wired24": {"bw": 24, "rtt": 20, "bdp_mult": 1, "bw_factor": 1},
            "wired48": {"bw": 48, "rtt": 20, "bdp_mult": 1, "bw_factor": 1},
            "wired96": {"bw": 96, "rtt": 20, "bdp_mult": 1, "bw_factor": 1},
            "wired6-2x": {"bw": 42, "rtt": 20, "bdp_mult": 1, "bw_factor": 2},
            "wired6-4x": {"bw": 42, "rtt": 20, "bdp_mult": 1, "bw_factor": 4},
            "wired6-8x": {"bw": 42, "rtt": 20, "bdp_mult": 1, "bw_factor": 8},
            "wired12-2x": {"bw": 42, "rtt": 20, "bdp_mult": 1, "bw_factor": 2},
            "wired12-4x": {"bw": 42, "rtt": 20, "bdp_mult": 1, "bw_factor": 4},
            "wired12-8x": {"bw": 42, "rtt": 20, "bdp_mult": 1, "bw_factor": 8},
            "wired24-2x": {"bw": 42, "rtt": 20, "bdp_mult": 1, "bw_factor": 2},
            "wired24-4x": {"bw": 42, "rtt": 20, "bdp_mult": 1, "bw_factor": 4},
            "wired24-8x": {"bw": 42, "rtt": 20, "bdp_mult": 1, "bw_factor": 8},
            "wired48-2x": {"bw": 42, "rtt": 20, "bdp_mult": 1, "bw_factor": 2},
            "wired48-4x": {"bw": 42, "rtt": 20, "bdp_mult": 1, "bw_factor": 4},
            "wired48-8x": {"bw": 42, "rtt": 20, "bdp_mult": 1, "bw_factor": 8},
            "wired96-2x": {"bw": 42, "rtt": 20, "bdp_mult": 1, "bw_factor": 2},
            "wired96-4x": {"bw": 42, "rtt": 20, "bdp_mult": 1, "bw_factor": 4},
            "wired96-8x": {"bw": 42, "rtt": 20, "bdp_mult": 1, "bw_factor": 8},
            "wired192-2x": {"bw": 42, "rtt": 20, "bdp_mult": 1, "bw_factor": 2},
            "wired192-4x": {"bw": 42, "rtt": 20, "bdp_mult": 1, "bw_factor": 4},
            "wired192-8x": {"bw": 42, "rtt": 20, "bdp_mult": 1, "bw_factor": 8},
        }
    
    @property
    def bw(self):
        return self.settings["bw"]
    
    @property
    def rtt(self):
        return self.settings["rtt"]
    
    @property
    def bdp_mult(self):
        return self.settings["bdp_mult"]
    
    @property
    def bw_factor(self):
        return self.settings["bw_factor"]
    
    def __str__(self):
        return self.name

