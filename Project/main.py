from Model_reglas_impuestas.Model import InitProgram
import pyodbc
import numpy as np
import pandas as pd
import os
import time
import pyarrow as pa
import pyarrow.parquet as pq
from multiprocessing.pool import ThreadPool as Pool
import json
import datetime

if __name__ == "__main__":
        print(np.__version__)
        #InitProgram()