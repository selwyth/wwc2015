#Introduction
We like to do snake drafts on major sporting events to force ourselves to better understand the sport and teams. This involves players picking teams in a snake-draft order (e.g. A-B-C-D-D-C-B-A-A-B-C-D and so forth) and scoring points for each win or tie recorded by each player's owned teams. For the 2015 FIFA Women's World Cup, I would like to validate or suggest my strategy by predicting tournament outcomes through a stochastic process.

The classic way would be to model goals scored for and against as separate Poisson distributions in a match, then use evidence gathered in previous games to construct posterior distributions for goals scored for and against for each match, as illustrated by Allen B. Downey in Think Bayes. But the folks at FiveThirtyEight [have already done this](http://projects.fivethirtyeight.com/womens-world-cup/) exhaustively in far greater detail than I can hope for, including accounting for extra time in the knockout stages, and updating team ratings iteratively with each simulated tournament to swamp the priors.

So I'll do something far simpler and different resembling a basic Elo rating system.

##Methodology
I build a gaussian distribution `np.normal(mu,sigma)` of performance for each team, with mean performance `mu` estimated either by FIFA ratings or FiveThirtyEight's WSPI ratings, and `sigma` estimated using the standard deviation of all team ratings in the Women's World Cup for convenience. With more effort, `sigma` can be tuned against the actual win-loss record between two teams.

Then for each match, I sample once from each team's performance distribution, and record it as a tie (after regulation) if the results fall within `z = np.sqrt(2) * sigma` of each other (two-sample t-test with equal variance), a win for whichever team has the higher rating if the difference is greater than `z`. `z` can be tuned against the actual rate of ties between teams.

By our rules, a win is awarded 2 points and a tie is awarded 1 point. These are only awarded for regulation, and the rules hold for knockout stages (e.g. if Team A beats Team B in extra time, both teams get 1 point, and Team A advances).

The process is simulated 1000 times. There is no iteration of results from each simulation into the next simulation, as I have not built any assumptions into or attempted to translate how match results and tournament performance update the performance distributions.

##Results
More info in normmodels.ipynb.
[!Score distribution using FTE](plots/scoredist.png)
[!Probability of winning](plots/winners.png)
[!Probability of getting to the semifinals](plots/top4.png)
[!Probability of getting out of the group stage](plots/round16.png)