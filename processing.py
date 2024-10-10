import numpy as np
import requests
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy.stats import norm

def get_replay(url):
    resp = requests.get(url)
    return resp.json()

def parse_replay(replay, filter=True):
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

    return filter_laps(laps) if filter else laps

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

def plot_norms(laps, alien_time):
    f, ax = plt.subplots()
    x_axis = np.arange(alien_time * 0.98, alien_time * 1.1, 0.05)
    m_v = mean_var_from_laps(laps)
    for driver, (m, std) in m_v.items():
        ax.plot(x_axis, norm.pdf(x_axis, m, std), label=driver)
    ax.axline((alien_time, 0), (alien_time, 1), linestyle="--", label="alien time")
    ax.axline((alien_time*1.03, 0), (alien_time*1.03, 1), linestyle=":", label="alien time + 3%")
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
    'bathhurst': {
        "2024": {
            "race": "https://www.accreplay.com/api/replays/31871",
            "qualifying": "",
            "practice": [],
        },
        "alien_time": 118.9,
    },
    'silverstone': {
        "2024": {
            "race": "https://www.accreplay.com/api/replays/31874",
            "practice": [],
        },
        "alien_time": 116.6,
    },
    'kyalami': {
        "2024": {
            "race": "https://www.accreplay.com/api/replays/32377",
            "practice": [],
        },
        "alien_time": 99.5,
    },
    'monza': {
        "2024": {
            "race": "https://www.accreplay.com/api/replays/32378",
            "practice": [],
        },
        "alien_time": 105.9,
    },
    'hungaroring': {
        "2024": {
             "race": "https://www.accreplay.com/api/replays/32379",
            "practice": [],
        },
        "alien_time": 102.2,
    },
    'zolder': {
        "2024": {
            "race": "https://www.accreplay.com/api/replays/32380",
            "practice": [],
        },
        "alien_time": 86.9,
    },
    'spa': {
        "2024": {
            "race": "https://www.accreplay.com/api/replays/32381",
         "practice": [],
        },
        "alien_time": 135.2,
    },
    'imola': {
        "2024": {
            "race": "https://www.accreplay.com/api/replays/32382",
            "practice": [],
        },
        "alien_time": 99.5,
    },
    'laguna_seca': {
        "2024": {
            "race": "https://www.accreplay.com/api/replays/32384",
            "practice": [],
        },
        "alien_time": 81.2,
    },
    'nurburgring_24h': {
        "2024": {
            "race": "https://www.accreplay.com/api/replays/32383",
            "practice": [],
        },
        "alien_time": 485.0,
    },
    'misano': {
        "2024": {
            "race": "https://www.accreplay.com/api/replays/32392",
            "practice": ["https://www.accreplay.com/api/replays/32464"],
            "qualifying": "https://www.accreplay.com/api/replays/32463",
        },
        "alien_time": 92.4,
    }
}

track = "imola"
laps = parse_replay(get_replay(REPLAYS[track]["2024"]["race"]), filter=True)
alien_time = REPLAYS[track]["alien_time"]
plot_max_speed(laps)
plot_norms(laps, alien_time)
plot_laps(laps)
plot_sorted_laps(laps)
plt.show()

