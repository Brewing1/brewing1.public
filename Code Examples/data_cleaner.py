
"""
This file creates a pandas dataframe from sports data located in a csv file.
The data is then cleaned for use in various other files.
""""

import pandas as pd
import os

class Clean(object):

    def __init__(self):
        self.dataset = pd.read_csv(os.path.join(os.path.dirname(__file__),
                                '../dat/odds_data2.csv'), error_bad_lines=False)

    def clean_data(self, odds=False):
        matches = self.dataset

        # add margin column
        matches.insert(6, 'Margin', matches['TeamScore1']-matches['TeamScore2'])

        # Gets rid of unwanted columns
        matches = matches.drop(['TeamScore1', 'TeamScore2'], axis=1)

        # Remove all columns containing odds information if the user doesn't
        # specifically request them
        if not odds:
            matches = matches.drop(['OddsOpen1','OddsOpen2','OddsClose1',
                    'OddsClose2','LineOpen1','LineClose1','LineOddsOpen1',
                    'LineOddsOpen2', 'LineOddsClose1','LineOddsClose2',], axis=1)

        # Replaces column-name "Finals Week" with "Final"
        matches.loc[matches.RoundType == 'Finals Week', 'RoundType'] = 'Final'

        # Gets rid of Preliminary Finals round 1 and replaces with Final round 3
        a = matches.loc[(matches.RoundType=='Preliminary Finals', 'RoundType')]
        for i in a.index:
            matches.loc[i,'RoundNo'] = 3
            matches.loc[i,'RoundType'] = 'Final'

        # Gets rid of Grand Final round 1 and replaces with Final round 4
        a = matches.loc[(matches.RoundType=='Grand Final', 'RoundType')]
        for i in a.index:
            matches.loc[i,'RoundNo'] = 4
            matches.loc[i,'RoundType'] = 'Final'


        # Starts the match index from 1 instead of 0
        matches.index += 1

        # order the data by season, RoundType, RoundNo
        matches = matches.sort_values(['MatchSeason', 'RoundType', 'RoundNo'],
            ascending=[True,False,True])

        return matches
