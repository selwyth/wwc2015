import numpy as np
from scipy import stats

import pandas as pd

df = pd.DataFrame({
    'team':['Germany','USA','France','Japan','Sweden','England','Brazil','Canada','Australia','Norway','Netherlands','Spain',
           'China','New Zealand','South Korea','Switzerland','Mexico','Colombia','Thailand','Nigeria','Ecuador','Ivory Coast','Cameroon','Costa Rica'],
    'group':['B','D','F','C','D','F','E','A','D','B','A','E','A','A','E','C','F','F','B','D','C','B','C','E'],
    'fifascore':[2168,2158,2103,2066,2008,2001,1984,1969,1968,1933,1919,1867,1847,1832,1830,1813,1748,1692,1651,1633,1485,1373,1455,1589],
    'ftescore':[95.6,95.4,92.4,92.7,91.6,89.6,92.2,90.1,88.7,88.7,86.2,84.7,85.2,82.5,84.3,83.7,81.1,78.0,68.0,85.7,63.3,75.6,79.3,72.8]
    })
	
df.index = df.team

import itertools

matchdf = pd.DataFrame({grp: tuple(itertools.combinations(team, 2)) for grp, team in df.groupby('group')['team']})
matchdf = pd.melt(matchdf, var_name='group')
matchdf['team1'] = matchdf.value.map(lambda x: x[0])
matchdf['team2'] = matchdf.value.map(lambda x: x[1])
matchdf.drop('value', axis=1, inplace=True)

def score_comparator(A, B, method):
    """ 2 pts per win, 1 pt per draw per our rules."""
    tscore = np.sqrt(np.power(2*df[method].std(),2))
    if abs(A-B) < tscore:
        return 1,1
    elif A - B >= tscore:
        return 2,0
    else:
        return 0,2

def rank_tb(series):
    """Ranks a series of scores and returns a list of ranks (higher is better), with ties broken randomly."""
    ranks = series.reindex(np.random.permutation(series.index))
    ranks = ranks.rank(method='first')
    return ranks
    
class mc(object):
    """ Manages the running of the simulations. 
    Instantiate with `method` in ['fifascore', 'ftescore'] and `simulations`, an int for number of trials.
    Currently requires data to be in proper format and properly named as `df` and `matchdf`
      
    Attributes:
        group_score: A DataFrame that contains group scores for each trial
        group_rank: A DataFrame that ranks teams (max method) in each trial
        total_score: A DataFrame that contains scores for each trial
        resultlist: A DataFrame that contains all match results, used to collate `totalscore`
    """
    def __init__(self, method, simulations):
        self.method = method
        self.simulations = simulations
    
    def run_sim(self):
        """Simulates the tournament `simulations` times.
        Returns a tuple (x,y) where x is a DataFrame of matches and y is a DataFrame of score for the last trial (to check).
        """
        scorelist = []
        rklist=[]
        firstlist=[]
        secondlist=[]
        thirdlist=[]
        tscorelist=[]
        resultlist=[]
        
        for i in range(self.simulations):
            matchdf['mean1'] = matchdf.team1.map(df[self.method]) # bring over the mean
            matchdf['mean2'] = matchdf.team2.map(df[self.method])
            matchdf['draw1'] = matchdf.mean1.map(lambda x: stats.norm(x, df[self.method].std()).rvs(1)[0]) # random pull from normdist
            matchdf['draw2'] = matchdf.mean2.map(lambda x: stats.norm(x, df[self.method].std()).rvs(1)[0])
            matchdf['score1'] = matchdf.apply(lambda x: score_comparator(x.draw1, x.draw2, self.method)[0], axis=1) # points for team1
            matchdf['score2'] = matchdf.apply(lambda x: score_comparator(x.draw1, x.draw2, self.method)[1], axis=1)
            scores = pd.concat([matchdf.groupby('team1').sum().score1, matchdf.groupby('team2').sum().score2],
                                axis=1, join='outer')
            scores.fillna(0, inplace=True)
            scores = scores.score1 + scores.score2
            scorelist.append(scores)
        
            ranks = scores.rank(method='max')
            rklist.append(ranks)
            
            grouprks = pd.concat({'score': scores, 'grp': df.group}, axis=1)
            grouprks = grouprks.groupby('grp').apply(lambda x: rank_tb(x.score))
            grouprks = pd.merge(grouprks.reset_index(), scores.reset_index(),
                                left_on='level_1', right_on='index')
            grouprks.drop(['index'], inplace=True, axis=1)
            grouprks.columns = ['grp', 'team', 'grouprk', 'score']
            first = grouprks[grouprks.grouprk==4]
            firstlist.append(first)
            second = grouprks[grouprks.grouprk==3]
            secondlist.append(second)
            
            thirds = grouprks[grouprks.grouprk==2]
            thirds.index=thirds.team
            thirds['grouprk'] = rank_tb(thirds.score)
            thirds.reset_index(inplace=True, drop=True)
            third = thirds[thirds.grouprk >= 3] # the top 4 3rd-placed finishers
            thirdlist.append(third)
        
            round16 = pd.DataFrame([
                {'id':37, 'team1':second.team[second.grp=='A'].iloc[0],'team2':second.team[second.grp=='C'].iloc[0]},
                {'id':40, 'team1':first.team[first.grp=='F'].iloc[0],'team2':second.team[second.grp=='E'].iloc[0]},
                {'id':41, 'team1':first.team[first.grp=='E'].iloc[0],'team2':second.team[second.grp=='D'].iloc[0]},
                {'id':43, 'team1':second.team[second.grp=='B'].iloc[0],'team2':second.team[second.grp=='F'].iloc[0]}
                ])
            
            if set(third.grp.tolist())==set(['A','B','C','D']):
                round16_2 = pd.DataFrame([
                    {'id':44, 'team1':first.team[first.grp=='A'].iloc[0], 'team2':thirds.team[thirds.grp=='C'].iloc[0]},
                    {'id':39, 'team1':first.team[first.grp=='B'].iloc[0], 'team2':thirds.team[thirds.grp=='D'].iloc[0]},
                    {'id':42, 'team1':first.team[first.grp=='C'].iloc[0], 'team2':thirds.team[thirds.grp=='A'].iloc[0]},
                    {'id':38, 'team1':first.team[first.grp=='D'].iloc[0], 'team2':thirds.team[thirds.grp=='B'].iloc[0]}
                    ])
            elif set(third.grp.tolist())==set(['A','B','C','E']):
                round16_2 = pd.DataFrame([
                    {'id':44, 'team1':first.team[first.grp=='A'].iloc[0], 'team2':thirds.team[thirds.grp=='C'].iloc[0]},
                    {'id':39, 'team1':first.team[first.grp=='B'].iloc[0], 'team2':thirds.team[thirds.grp=='A'].iloc[0]},
                    {'id':42, 'team1':first.team[first.grp=='C'].iloc[0], 'team2':thirds.team[thirds.grp=='B'].iloc[0]},
                    {'id':38, 'team1':first.team[first.grp=='D'].iloc[0], 'team2':thirds.team[thirds.grp=='E'].iloc[0]}
                    ])
            elif set(third.grp.tolist())==set(['A','B','C','F']):
                round16_2 = pd.DataFrame([
                    {'id':44, 'team1':first.team[first.grp=='A'].iloc[0], 'team2':thirds.team[thirds.grp=='C'].iloc[0]},
                    {'id':39, 'team1':first.team[first.grp=='B'].iloc[0], 'team2':thirds.team[thirds.grp=='A'].iloc[0]},
                    {'id':42, 'team1':first.team[first.grp=='C'].iloc[0], 'team2':thirds.team[thirds.grp=='B'].iloc[0]},
                    {'id':38, 'team1':first.team[first.grp=='D'].iloc[0], 'team2':thirds.team[thirds.grp=='F'].iloc[0]}
                    ])
            elif set(third.grp.tolist())==set(['A','B','D','E']):
                round16_2 = pd.DataFrame([
                    {'id':44, 'team1':first.team[first.grp=='A'].iloc[0], 'team2':thirds.team[thirds.grp=='D'].iloc[0]},
                    {'id':39, 'team1':first.team[first.grp=='B'].iloc[0], 'team2':thirds.team[thirds.grp=='A'].iloc[0]},
                    {'id':42, 'team1':first.team[first.grp=='C'].iloc[0], 'team2':thirds.team[thirds.grp=='B'].iloc[0]},
                    {'id':38, 'team1':first.team[first.grp=='D'].iloc[0], 'team2':thirds.team[thirds.grp=='E'].iloc[0]}
                    ])
            elif set(third.grp.tolist())==set(['A','B','D','F']):
                round16_2 = pd.DataFrame([
                    {'id':44, 'team1':first.team[first.grp=='A'].iloc[0], 'team2':thirds.team[thirds.grp=='D'].iloc[0]},
                    {'id':39, 'team1':first.team[first.grp=='B'].iloc[0], 'team2':thirds.team[thirds.grp=='A'].iloc[0]},
                    {'id':42, 'team1':first.team[first.grp=='C'].iloc[0], 'team2':thirds.team[thirds.grp=='B'].iloc[0]},
                    {'id':38, 'team1':first.team[first.grp=='D'].iloc[0], 'team2':thirds.team[thirds.grp=='F'].iloc[0]}
                    ])
            elif set(third.grp.tolist())==set(['A','B','E','F']):
                round16_2 = pd.DataFrame([
                    {'id':44, 'team1':first.team[first.grp=='A'].iloc[0], 'team2':thirds.team[thirds.grp=='E'].iloc[0]},
                    {'id':39, 'team1':first.team[first.grp=='B'].iloc[0], 'team2':thirds.team[thirds.grp=='A'].iloc[0]},
                    {'id':42, 'team1':first.team[first.grp=='C'].iloc[0], 'team2':thirds.team[thirds.grp=='B'].iloc[0]},
                    {'id':38, 'team1':first.team[first.grp=='D'].iloc[0], 'team2':thirds.team[thirds.grp=='F'].iloc[0]}
                    ])
            elif set(third.grp.tolist())==set(['A','C','D','E']):
                round16_2 = pd.DataFrame([
                    {'id':44, 'team1':first.team[first.grp=='A'].iloc[0], 'team2':thirds.team[thirds.grp=='C'].iloc[0]},
                    {'id':39, 'team1':first.team[first.grp=='B'].iloc[0], 'team2':thirds.team[thirds.grp=='D'].iloc[0]},
                    {'id':42, 'team1':first.team[first.grp=='C'].iloc[0], 'team2':thirds.team[thirds.grp=='A'].iloc[0]},
                    {'id':38, 'team1':first.team[first.grp=='D'].iloc[0], 'team2':thirds.team[thirds.grp=='E'].iloc[0]}
                    ])
            elif set(third.grp.tolist())==set(['A','C','D','F']):
                round16_2 = pd.DataFrame([
                    {'id':44, 'team1':first.team[first.grp=='A'].iloc[0], 'team2':thirds.team[thirds.grp=='C'].iloc[0]},
                    {'id':39, 'team1':first.team[first.grp=='B'].iloc[0], 'team2':thirds.team[thirds.grp=='D'].iloc[0]},
                    {'id':42, 'team1':first.team[first.grp=='C'].iloc[0], 'team2':thirds.team[thirds.grp=='A'].iloc[0]},
                    {'id':38, 'team1':first.team[first.grp=='D'].iloc[0], 'team2':thirds.team[thirds.grp=='F'].iloc[0]}
                    ])
            elif set(third.grp.tolist())==set(['A','C','E','F']):
                round16_2 = pd.DataFrame([
                    {'id':44, 'team1':first.team[first.grp=='A'].iloc[0], 'team2':thirds.team[thirds.grp=='C'].iloc[0]},
                    {'id':39, 'team1':first.team[first.grp=='B'].iloc[0], 'team2':thirds.team[thirds.grp=='D'].iloc[0]},
                    {'id':42, 'team1':first.team[first.grp=='C'].iloc[0], 'team2':thirds.team[thirds.grp=='F'].iloc[0]},
                    {'id':38, 'team1':first.team[first.grp=='D'].iloc[0], 'team2':thirds.team[thirds.grp=='E'].iloc[0]}
                    ])
            elif set(third.grp.tolist())==set(['A','D','E','F']):
                round16_2 = pd.DataFrame([
                    {'id':44, 'team1':first.team[first.grp=='A'].iloc[0], 'team2':thirds.team[thirds.grp=='D'].iloc[0]},
                    {'id':39, 'team1':first.team[first.grp=='B'].iloc[0], 'team2':thirds.team[thirds.grp=='A'].iloc[0]},
                    {'id':42, 'team1':first.team[first.grp=='C'].iloc[0], 'team2':thirds.team[thirds.grp=='F'].iloc[0]},
                    {'id':38, 'team1':first.team[first.grp=='D'].iloc[0], 'team2':thirds.team[thirds.grp=='E'].iloc[0]}
                    ])
            elif set(third.grp.tolist())==set(['B','C','D','E']):
                round16_2 = pd.DataFrame([
                    {'id':44, 'team1':first.team[first.grp=='A'].iloc[0], 'team2':thirds.team[thirds.grp=='C'].iloc[0]},
                    {'id':39, 'team1':first.team[first.grp=='B'].iloc[0], 'team2':thirds.team[thirds.grp=='D'].iloc[0]},
                    {'id':42, 'team1':first.team[first.grp=='C'].iloc[0], 'team2':thirds.team[thirds.grp=='B'].iloc[0]},
                    {'id':38, 'team1':first.team[first.grp=='D'].iloc[0], 'team2':thirds.team[thirds.grp=='E'].iloc[0]}
                    ])
            elif set(third.grp.tolist())==set(['B','C','D','F']):
                round16_2 = pd.DataFrame([
                    {'id':44, 'team1':first.team[first.grp=='A'].iloc[0], 'team2':thirds.team[thirds.grp=='C'].iloc[0]},
                    {'id':39, 'team1':first.team[first.grp=='B'].iloc[0], 'team2':thirds.team[thirds.grp=='D'].iloc[0]},
                    {'id':42, 'team1':first.team[first.grp=='C'].iloc[0], 'team2':thirds.team[thirds.grp=='B'].iloc[0]},
                    {'id':38, 'team1':first.team[first.grp=='D'].iloc[0], 'team2':thirds.team[thirds.grp=='F'].iloc[0]}
                    ])
            elif set(third.grp.tolist())==set(['B','C','E','F']):
                round16_2 = pd.DataFrame([
                    {'id':44, 'team1':first.team[first.grp=='A'].iloc[0], 'team2':thirds.team[thirds.grp=='E'].iloc[0]},
                    {'id':39, 'team1':first.team[first.grp=='B'].iloc[0], 'team2':thirds.team[thirds.grp=='C'].iloc[0]},
                    {'id':42, 'team1':first.team[first.grp=='C'].iloc[0], 'team2':thirds.team[thirds.grp=='B'].iloc[0]},
                    {'id':38, 'team1':first.team[first.grp=='D'].iloc[0], 'team2':thirds.team[thirds.grp=='F'].iloc[0]}
                    ])
            elif set(third.grp.tolist())==set(['B','D','E','F']):
                round16_2 = pd.DataFrame([
                    {'id':44, 'team1':first.team[first.grp=='A'].iloc[0], 'team2':thirds.team[thirds.grp=='E'].iloc[0]},
                    {'id':39, 'team1':first.team[first.grp=='B'].iloc[0], 'team2':thirds.team[thirds.grp=='D'].iloc[0]},
                    {'id':42, 'team1':first.team[first.grp=='C'].iloc[0], 'team2':thirds.team[thirds.grp=='B'].iloc[0]},
                    {'id':38, 'team1':first.team[first.grp=='D'].iloc[0], 'team2':thirds.team[thirds.grp=='F'].iloc[0]}
                    ])
            elif set(third.grp.tolist())==set(['C','D','E','F']):
                round16_2 = pd.DataFrame([
                    {'id':44, 'team1':first.team[first.grp=='A'].iloc[0], 'team2':thirds.team[thirds.grp=='C'].iloc[0]},
                    {'id':39, 'team1':first.team[first.grp=='B'].iloc[0], 'team2':thirds.team[thirds.grp=='D'].iloc[0]},
                    {'id':42, 'team1':first.team[first.grp=='C'].iloc[0], 'team2':thirds.team[thirds.grp=='F'].iloc[0]},
                    {'id':38, 'team1':first.team[first.grp=='D'].iloc[0], 'team2':thirds.team[thirds.grp=='E'].iloc[0]}
                    ])
            else:
                round16_2 = None
                print "This shouldn't happen"
                
            round16f = pd.concat([round16,round16_2], ignore_index=True)
            round16f.set_index('id', drop=True, inplace=True)
            
            # Simulate the round of 16
            round16f['mean1'] = round16f.team1.map(df[self.method])
            round16f['mean2'] = round16f.team2.map(df[self.method])
            round16f['draw1'] = round16f.mean1.map(lambda x: stats.norm(x, df[self.method].std()).rvs(1)[0])
            round16f['draw2'] = round16f.mean2.map(lambda x: stats.norm(x, df[self.method].std()).rvs(1)[0])
            round16f['score1'] = round16f.apply(lambda x: score_comparator(x.draw1, x.draw2, self.method)[0], axis=1) # points for team1
            round16f['score2'] = round16f.apply(lambda x: score_comparator(x.draw1, x.draw2, self.method)[1], axis=1)
            round16f['winner'] = round16f.apply(lambda x: x.team1 if x.draw1 > x.draw2 else x.team2, axis=1)
            
            # Generate the quarter-final matches
            qtrs = pd.DataFrame([
                {'id':45, 'team1':round16f.loc[37].winner, 'team2':round16f.loc[38].winner},
                {'id':46, 'team1':round16f.loc[39].winner, 'team2':round16f.loc[40].winner},
                {'id':47, 'team1':round16f.loc[41].winner, 'team2':round16f.loc[42].winner},
                {'id':48, 'team1':round16f.loc[43].winner, 'team2':round16f.loc[44].winner}
                ])
            qtrs.set_index('id', drop=True, inplace=True)
            
            # Simulate the quarter-final matches
            qtrs['mean1'] = qtrs.team1.map(df[self.method])
            qtrs['mean2'] = qtrs.team2.map(df[self.method])
            qtrs['draw1'] = qtrs.mean1.map(lambda x: stats.norm(x, df[self.method].std()).rvs(1)[0])
            qtrs['draw2'] = qtrs.mean2.map(lambda x: stats.norm(x, df[self.method].std()).rvs(1)[0])
            qtrs['score1'] = qtrs.apply(lambda x: score_comparator(x.draw1, x.draw2, self.method)[0], axis=1) # points for team1
            qtrs['score2'] = qtrs.apply(lambda x: score_comparator(x.draw1, x.draw2, self.method)[1], axis=1)
            qtrs['winner'] = qtrs.apply(lambda x: x.team1 if x.draw1 > x.draw2 else x.team2, axis=1)
            
            # Generate the semifinal matches
            semis = pd.DataFrame([
                {'id':49, 'team1':qtrs.loc[45].winner, 'team2':qtrs.loc[46].winner},
                {'id':50, 'team1':qtrs.loc[47].winner, 'team2':qtrs.loc[48].winner}])
            semis.set_index('id', drop=True, inplace=True)
            
            # Simulate the semi-finals
            semis['mean1'] = semis.team1.map(df[self.method])
            semis['mean2'] = semis.team2.map(df[self.method])
            semis['draw1'] = semis.mean1.map(lambda x: stats.norm(x, df[self.method].std()).rvs(1)[0])
            semis['draw2'] = semis.mean2.map(lambda x: stats.norm(x, df[self.method].std()).rvs(1)[0])
            semis['score1'] = semis.apply(lambda x: score_comparator(x.draw1, x.draw2, self.method)[0], axis=1) # points for team1
            semis['score2'] = semis.apply(lambda x: score_comparator(x.draw1, x.draw2, self.method)[1], axis=1)
            semis['winner'] = semis.apply(lambda x: x.team1 if x.draw1 > x.draw2 else x.team2, axis=1)
            semis['loser'] = semis.apply(lambda x: x.team1 if x.draw1 < x.draw2 else x.team2, axis=1)
                    
            # Generate the finals
            finals = pd.DataFrame([
                {'id':51, 'team1':semis.loc[49].loser, 'team2':semis.loc[50].loser},
                {'id':52, 'team1':semis.loc[49].winner, 'team2':semis.loc[50].winner}])
            finals.set_index('id', drop=True, inplace=True)
            
            # Simulate the finals
            finals['mean1'] = finals.team1.map(df[self.method])
            finals['mean2'] = finals.team2.map(df[self.method])
            finals['draw1'] = finals.mean1.map(lambda x: stats.norm(x, df[self.method].std()).rvs(1)[0])
            finals['draw2'] = finals.mean2.map(lambda x: stats.norm(x, df[self.method].std()).rvs(1)[0])
            finals['score1'] = finals.apply(lambda x: score_comparator(x.draw1, x.draw2, self.method)[0], axis=1) # points for team1
            finals['score2'] = finals.apply(lambda x: score_comparator(x.draw1, x.draw2, self.method)[1], axis=1)
            finals['winner'] = finals.apply(lambda x: x.team1 if x.draw1 > x.draw2 else x.team2, axis=1)
            
            # Consolidate scoring results
            results = pd.concat([matchdf, round16f, qtrs, semis, finals], axis=0)
            resultlist.append(results)
            
            totalscore = pd.concat([results.groupby('team1').sum().score1, results.groupby('team2').sum().score2],
                                   axis=1, join='outer')
            totalscore.fillna(0, inplace=True)
            totalscore = totalscore.score1 + totalscore.score2
            tscorelist.append(totalscore)
        
        self.group_score = pd.concat(scorelist, axis=1)
        self.group_rank = pd.concat(rklist, axis=1)
        self.total_score = tscorelist
        self.resultlist = resultlist
        return results, totalscore
    
if __name__ == "__main__":
    print 'test'