#!/usr/bin/env python3.9.13

#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

__maintainer__ = 'Rob Hoelzle'
__script_name__ = 'mssql_df_upsert.py'
__version__ = '1.0.0'
__profiling__ = 'False'

###############################################################################
###############################################################################

## Import libraries

import pandas as pd
import sqlalchemy as sa

###############################################################################
###############################################################################

## Main
#upsert
def df_upsert(data_frame: pd.DataFrame, table_name: str, engine: sa.engine.Engine,
              schema: str = None, match_columns: list = None):
    """
    Perform an "upsert" on a SQL Server table from a DataFrame.
    Constructs a T-SQL MERGE statement, uploads the DataFrame to a
    temporary table, and then executes the MERGE.
    
    Args:
    ----------
    data_frame : pandas.DataFrame
        The DataFrame to be upserted
    table_name : str
        The name of the target table
    engine : sqlalchemy.engine.Engine
        The SQLAlchemy Engine to use
    schema : str, optional
        The name of the schema containing the target table
    match_columns : list of str, optional
        A list of the column name(s) on which to match. If omitted, the
        primary key columns of the target table will be used
    
    Returns:
    ----------
    NA, upserts to sql table
    
    Raises:
    ----------
    
    """
    #define table spec based on schema and table name
    table_spec = ""
    if schema:
        table_spec += schema + "."
    table_spec += table_name

    #create list of match columns
    df_columns = list(data_frame.columns)
    if not match_columns:
        insp = sa.inspect(engine)
        match_columns = insp.get_pk_constraint(table_name, schema=schema)[
            "constrained_columns"
        ]
    
    #create list of update columns
    columns_to_update = [col for col in df_columns if col not in match_columns]
    
    #build upsert statement
    join_condition = " AND ".join([f"main.[{col}] = temp.[{col}]" for col in match_columns])
    update_list = ", ".join([f"[{col}] = temp.[{col}]" for col in columns_to_update])
    insert_cols_str = ", ".join([f"[{col}]" for col in df_columns])
    insert_vals_str = ", ".join([f"temp.[{col}]" for col in df_columns])

    stmt = f"""
        merge {table_spec} with (HOLDLOCK) as main
        using (select {', '.join([f'[{col}]' for col in df_columns])} from #temp_table) as temp
        on ({join_condition})
        when matched then
            update set {update_list}
        when not matched then
            insert ({insert_cols_str})
            values ({insert_vals_str});
            """

    #execute upsert statement
    with engine.connect() as cnxn:
        data_frame.to_sql("#temp_table", cnxn, index=False)
        cnxn.exec_driver_sql(stmt)
        cnxn.exec_driver_sql("DROP TABLE IF EXISTS #temp_table")