import numpy as np
import requests
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy.stats import norm

def get_replay(url):
    resp = requests.get(url)
    return resp.json()

def parse_replay(replay):
    cars = {}
    for car in replay["cars"]:
        cars[car['id']] = car['carModelName']
    drivers = {}
    for driver in replay["drivers"]:
        drivers[driver['id']] = f"{driver["firstName"]} {driver['lastName']}: {cars[driver['carId']]}"

    laps = defaultdict(list)
    for lap in replay['laps']:
        if not lap or 'topSpeedKMH' not in lap:
            continue
        laps[drivers[lap['driverId']]].append((lap['lapNumber'], lap['lapTimeMS']/1000, lap['isValid'], lap['topSpeedKMH']))

    return filter_laps(laps)

def mean_var_from_laps(laps):
    ret = {}
    for driver, lap_l in laps.items():
        ret[driver] = np.mean([l[1] for l in lap_l]), np.std([l[1] for l in lap_l])
    return ret

def filter_laps(laps):
    ret = {}
    for driver, lap_l in laps.items():
        ret[driver] = sorted(sorted(lap_l, key=lambda x: x[1])[:-3], key=lambda x: x[0])
        if len(ret[driver]) == 0:
            del ret[driver]
    return ret

def plot_norms(laps, centre_time):
    f, ax = plt.subplots()
    x_axis = np.arange(centre_time * 0.9, centre_time * 1.1, 0.05)
    m_v = mean_var_from_laps(laps)
    for driver, (m, std) in m_v.items():
        ax.plot(x_axis, norm.pdf(x_axis, m, std), label=driver)
    ax.set_title("Lap time distribution")
    ax.set_xlabel("Lap time (s)")
    ax.legend()

def plot_laps(laps):
    f, ax = plt.subplots()
    for d, l in laps.items():
        plt.plot([lap[0] for lap in l], [lap[1] for lap in l], label=d)
    ax.set_title("Laps in order")
    ax.set_ylabel("Lap time (s)")
    ax.set_xlabel("Lap of race")
    ax.legend()

def plot_sorted_laps(laps):
    f, ax = plt.subplots()
    for d, l in laps.items():
        l = sorted(l, key=lambda x: x[1])
        ax.plot(list(range(len(l))), [lap[1] for lap in l], label=d)
    ax.set_title("Laps by rank")
    ax.set_ylabel("Lap time (s)")
    ax.set_xlabel("Lap rank")
    ax.legend()

def plot_max_speed(laps):
    f, ax = plt.subplots()
    maxes = [sorted(l, key=lambda x: x[3])[-1][3] for l in laps.values()]
    ax.bar([d for d in laps.keys()], maxes)
    ax.grid(True)
    ax.set_title("Max Speed")
    ax.set_ylabel("Max Speed (KPH)")
    ax.set_ylim([min(maxes)*0.98, max(maxes)*1.02])
    plt.xticks(rotation=30, fontsize=8)

REPLAYS = {
    'bathhurst': ("https://www.accreplay.com/api/replays/31871", 124),
    'silverstone': ("https://www.accreplay.com/api/replays/31874", 122),
    'kyalami': ("https://www.accreplay.com/api/replays/32377", 104),
    'monza': ("https://www.accreplay.com/api/replays/32378", 110),
    'hungaroring': ("https://www.accreplay.com/api/replays/32379", 107),
    'zolder': ("https://www.accreplay.com/api/replays/32380", 91),
    'spa': ("https://www.accreplay.com/api/replays/32381", 145),
    'imola': ("https://www.accreplay.com/api/replays/32382", 105),
    'laguna_seca': ("https://www.accreplay.com/api/replays/32384", 90),
    'nurburgring_24h': ("https://www.accreplay.com/api/replays/32383", 520),
    'misano': ("https://www.accreplay.com/api/replays/32392", 100),
}

laps = parse_replay(get_replay(REPLAYS["bathhurst"][0]))
centre_time = REPLAYS["bathhurst"][1]
plot_max_speed(laps)
plot_norms(laps, centre_time)
plot_laps(laps)
plot_sorted_laps(laps)
plt.show()

