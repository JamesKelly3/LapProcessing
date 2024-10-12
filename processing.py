import numpy as np
import requests
from collections import defaultdict
import matplotlib.pyplot as plt
from PIL.ImageOps import scale
from scipy.stats import norm
import textwrap
import json

def get_replay(url):
    resp = requests.get(url)
    return resp.json()

def parse_replay(replay, filter=True):
    cars = {}
    for car in replay["cars"]:
        cars[car['id']] = car['carModelName']
    drivers = {}
    for driver in replay["drivers"]:
        drivers[driver['id']] = f"{driver["firstName"]} {driver['lastName']}: {cars[driver['carId']]}", driver["shortName"]

    laps = defaultdict(list)
    for lap in replay['laps']:
        if not lap or 'topSpeedKMH' not in lap:
            continue
        laps[drivers[lap['driverId']]].append((lap['lapNumber'], lap['lapTimeMS']/1000, lap['isValid'], lap['topSpeedKMH']))

    return filter_laps(laps) if filter else laps

def parse_acc_rc_dump(file):
    with open(file) as acc_rc_dump:
        parsed = json.load(acc_rc_dump)
        pit_times = defaultdict(dict)
        valid_laps = defaultdict(list)
        accidents = []
        sector_times = defaultdict(lambda: defaultdict(lambda: float("inf")))
        for lap in parsed["laps"]:
            if lap["pitTime"] > 10 * 1000: # file sometimes has tiny pitstops (I guess a bug in acc_rc) *1000 because it's ms
                pit_times[lap["driverNickName"]][lap["lapNumber"]] = lap["pitTime"]
            valid_laps[lap["driverNickName"]].append(lap["flags"] != 0)
            if lap["accidents"] in (1, 1025):
                accidents.append((lap["driverNickName"], lap["lapNumber"]))
            if lap["flags"] == 0:
                for sector in ('sector1', 'sector2', 'sector3'):
                    driver = lap['driverNickName']
                    sector_times[driver][sector] = min(sector_times[driver][sector], lap[sector]/1000)
    return pit_times, valid_laps, accidents, sector_times


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
    for (driver, shortName), (m, std) in m_v.items():
        ax.plot(x_axis, norm.pdf(x_axis, m, std), label=driver, color=DRIVER_COLOURS[shortName])
    ax.axline((alien_time, 0), (alien_time, 1), linestyle="--", label="alien time", )
    ax.axline((alien_time*1.03, 0), (alien_time*1.03, 1), linestyle=":", label="alien time + 3%")
    ax.set_title("Lap time distribution")
    ax.set_xlabel("Lap time (s)")
    ax.legend()

def plot_laps(laps):
    f, ax = plt.subplots()
    for (d, s), l in laps.items():
        plt.plot([lap[0] for lap in l], [lap[1] for lap in l], label=d, color=DRIVER_COLOURS[s])
    ax.set_title("Laps in order")
    ax.set_ylabel("Lap time (s)")
    ax.set_xlabel("Lap of race")
    ax.legend()

def format_time(s):
    return '{:02}:{:.2f}'.format(round(s) % 3600 // 60, s % 60)

def plot_sorted_laps(laps, alien_time):
    f, ax = plt.subplots()
    for (d, s), l in laps.items():
        l = sorted(l, key=lambda x: x[1])
        ax.plot(list(range(len(l))), [(lap[1]/alien_time) * 100 for lap in l], label=d, color=DRIVER_COLOURS[s])
    ax.axline((0, 103), (1, 103), linestyle=":", label="103%")
    ax.set_title("Laps by rank")
    ax.set_ylabel(f"pct of alien time ({format_time(alien_time)})")
    ax.set_xlabel("Lap rank")
    ax.legend()

def plot_max_speed(laps):
    f, ax = plt.subplots(tight_layout=True)
    f.set_figwidth(15)
    maxes = [np.mean([lap[3] for lap in l]) for l in laps.values()]
    x = [textwrap.fill(d, width=15) for (d, _s) in laps.keys()]
    ax.bar(x, maxes, color=[DRIVER_COLOURS[s] for (_d, s) in laps.keys()])
    for i in range(len(x)):
        plt.text(i, maxes[i], f"{maxes[i]:.2f}", ha='center')
    ax.grid(True)
    ax.set_title("Max Speed")
    ax.set_ylabel("Max Speed (KPH)")
    ax.set_ylim([min(maxes)*0.98, max(maxes)*1.02])
    plt.xticks(rotation=0, fontsize=8)

def plot_pit_times(pit):
    f, ax = plt.subplots(tight_layout=True)
    f.set_figwidth(15)
    i = 0
    for driver, times in pit.items():
        for lap_no, time_ms in times.items():
            ax.bar(f"{driver} Lap: {lap_no}", time_ms/1000, color=DRIVER_COLOURS[driver])
            plt.text(i, time_ms/1000, f"{time_ms/1000}", ha='center')
            i+=1
    ax.set_title("Pit stop times")

def plot_valid_laps(invalid):
    f, ax = plt.subplots(tight_layout=True)
    f.set_figwidth(15)
    i = 0
    for driver, laps_l in invalid.items():
        valid_laps = len([l for l in laps_l if not l])
        invalid_laps = len([l for l in laps_l if l])
        ax.bar(driver, valid_laps, color="green")
        ax.bar(driver, invalid_laps, bottom = valid_laps, color='red')
        plt.text(i, valid_laps+invalid_laps, f"{(valid_laps/(valid_laps+invalid_laps))*100:.2f}%", ha='center')
        i+=1
    ax.set_title('Valid laps')

def plot_accidents(accidents):
    f, ax = plt.subplots()
    for driver, lap in accidents:
        ax.scatter(lap, driver, color=DRIVER_COLOURS[driver],marker='x', s=100)
    plt.xticks(np.arange(max([l+1 for d, l in accidents])))
    ax.set_title("Accidents")

def plot_best_sectors(sector_times):
    f, ax = plt.subplots(2, 2)
    set_lim = defaultdict(lambda: False)
    for driver, sectors in sector_times.items():

        best_possible_time = sum(s for s in sectors.values())
        for sector, best_time in sectors.items():
            s_num = int(sector[-1]) -1
            ax[int(s_num/2)][int(s_num % 2)].bar(driver, best_time, color=DRIVER_COLOURS[driver])
            ax[int(s_num / 2)][int(s_num % 2)].set_title(f"Best {sector} time")
            new_ymin = best_time-0.2
            new_ymax = best_time+0.2
            if set_lim[sector]:
                ymin, ymax = ax[int(s_num / 2)][int(s_num % 2)].get_ylim()
                new_ymin = min(new_ymin, ymin)
                new_ymax = max(new_ymax, ymax)
            ax[int(s_num / 2)][int(s_num % 2)].set_ylim(new_ymin, new_ymax)
            set_lim[sector] = True

        new_ymin = best_possible_time - 0.2
        new_ymax = best_possible_time + 0.2
        if set_lim["overall"]:
            ymin, ymax = ax[1][1].get_ylim()
            new_ymin = min(new_ymin, ymin)
            new_ymax = max(new_ymax, ymax)
        ax[1][1].bar(driver, best_possible_time, color=DRIVER_COLOURS[driver])
        ax[1][1].set_title("Best possible time")
        ax[1][1].set_ylim(new_ymin, new_ymax)
        set_lim["overall"] = True

def plot_gaps(laps):
    f, ax = plt.subplots()
    cum_lap_time = {}
    for driver, lap_l in laps.items():
        out = [0]
        for i, lap in enumerate(lap_l):
            out.append(lap[1] + out[i])
        cum_lap_time[driver] = out
    for (driver, shortName), lap_l in cum_lap_time.items():
        out2 = []
        for i, cum_time in enumerate(lap_l):
            gap_to_first = max(cum_time - x[i] for x in cum_lap_time.values() if i < len(x))
            out2.append(gap_to_first)
        ax.plot(list(range(len(out2))), out2, label=driver, color=DRIVER_COLOURS[shortName])
    ax.set_title("Gap to first")
    ax.legend()
    ax.set_ylabel("Gap to first (s)")
    ax.set_xlabel("Lap")

DRIVER_COLOURS = {
    "CRE": "black",
    "RID": "indianred",
    "VLK": "darkgoldenrod",
    "DIV": "darkolivegreen",
    "WOJ": "springgreen",
    "RJT": "teal",
    "JKE": "darkorchid",
    "IYK": "red",
}

REPLAYS = {
    'bathurst': {
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
    },
    "cota": {
        "2024":{
            "race": "",
            "qualifying": "",
        },
        "alien_time": 124.5,
    },
}

plt.rcParams['figure.figsize'] = [11, 8]
track = "misano"
year = "2024"
session = "race"
laps = parse_replay(get_replay(REPLAYS[track][year][session]), filter=True)
unfiltered_laps = parse_replay(get_replay(REPLAYS[track][year][session]), filter=False)
pit, invalid, accidents, sector_times = parse_acc_rc_dump(f"ACC_companion_dumps/{track}.json")
alien_time = REPLAYS[track]["alien_time"]
plot_max_speed(laps)
plot_norms(laps, alien_time)
plot_laps(laps)
plot_sorted_laps(laps, alien_time)
plot_gaps(unfiltered_laps)
plot_pit_times(pit)
plot_valid_laps(invalid)
plot_accidents(accidents)
plot_best_sectors(sector_times)
plt.show()

