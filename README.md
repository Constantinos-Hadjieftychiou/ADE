# ADE

phases:
[
  {
    "name": "cold-start-trickle",   // Small trickle of traffic to warm up the service
    "pattern": "steady",            // Constant inter-arrival time
    "duration": 60,                 // 60 seconds
    "rps": 1                        // 1 request per second
  },
  {
    "name": "small-bursty-traffic", // Moderate bursts with gaps to simulate sporadic users
    "pattern": "burst",
    "duration": 180,                // 3 minutes total for this phase
    "rps": 8,                       // Average requests per second during bursts
    "burst": 25,                    // Mean burst duration (seconds)
    "idle": 20                      // Mean idle duration (seconds) between bursts
  },
  {
    "name": "mid-peak-poisson",     // Random arrivals like a busy period with jitter
    "pattern": "poisson",
    "duration": 300,                // 5 minutes
    "rps": 20                       // Average of 20 requests per second
  },
  {
    "name": "choppy-afternoon",     // Long, choppy bursts like afternoon traffic spikes
    "pattern": "burst",
    "duration": 240,                // 4 minutes
    "rps": 12,                      // Average RPS during bursts
    "burst": 40,                    // Longer burst periods
    "idle": 30                      // Longer idle periods between bursts
  },
  {
    "name": "long-idle",            // Extended idle window (no requests at all)
    "pattern": "steady",
    "duration": 180,                // 3 minutes
    "rps": 0                        // 0 RPS = pure idle in your client.py scheduling
  },
  {
    "name": "evening-cooldown",     // Light trickle of traffic as system cools down
    "pattern": "steady",
    "duration": 120,                // 2 minutes
    "rps": 3                        // Low steady background load
  }
]
