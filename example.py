import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
print(sys.path)
import time
import json
import pathlib
import coloredlogs
import logging
import matplotlib
import matplotlib.pyplot as plt

from itertools import zip_longest
import functools
import math
import logging
import numba
import enum
import multiprocessing as mp

import tqdm

import math
import itertools
import logging
import numpy as np




# switch matplotlib backend for running in background
matplotlib.use('agg')
matplotlib.rcParams['xtick.labelsize'] = '12'
matplotlib.rcParams['ytick.labelsize'] = '12'

coloredlogs.install('INFO', fmt='%(asctime)s [0x%(process)x] %(levelname)s %(message)s')
logger = logging.getLogger(__name__)





#Algorithms 


def _hamming_distance(result1, result2):
    # implement hamming distance in pure python, faster than np.count_zeros if inputs are plain python list
    return sum(res1 != res2 for res1, res2 in zip_longest(result1, result2))


def noisy_max_v1a(prng, queries, epsilon):
    # find the largest noisy element and return its index
    return (np.asarray(queries, dtype=np.float64) + prng.laplace(scale=2.0 / epsilon, size=len(queries))).argmax()


def noisy_max_v1b(prng, queries, epsilon):
    # INCORRECT: returning maximum value instead of the index
    return (np.asarray(queries, dtype=np.float64) + prng.laplace(scale=2.0 / epsilon, size=len(queries))).max()


def noisy_max_v2a(prng, queries, epsilon):
    return (np.asarray(queries, dtype=np.float64) + prng.exponential(scale=2.0 / epsilon, size=len(queries))).argmax()


def noisy_max_v2b(prng, queries, epsilon):
    # INCORRECT: returning the maximum value instead of the index
    return (np.asarray(queries, dtype=np.float64) + prng.exponential(scale=2.0 / epsilon, size=len(queries))).max()


def histogram_eps(prng, queries, epsilon):
    # INCORRECT: using (epsilon) noise instead of (1 / epsilon)
    noisy_array = np.asarray(queries, dtype=np.float64) + prng.laplace(scale=epsilon, size=len(queries))
    return noisy_array[0]


def histogram(prng, queries, epsilon):
    noisy_array = np.asarray(queries, dtype=np.float64) + prng.laplace(scale=1.0 / epsilon, size=len(queries))
    return noisy_array[0]


def SVT(prng, queries, epsilon, N, T):
    out = []
    eta1 = prng.laplace(scale=2.0 / epsilon)
    noisy_T = T + eta1
    c1 = 0
    for query in queries:
        eta2 = prng.laplace(scale=4.0 * N / epsilon)
        if query + eta2 >= noisy_T:
            out.append(True)
            c1 += 1
            if c1 >= N:
                break
        else:
            out.append(False)
    return out.count(False)


#Database Generation 



class Sensitivity(enum.Enum):
    ALL_DIFFER = 0
    ONE_DIFFER = 1


ALL_DIFFER = Sensitivity.ALL_DIFFER
ONE_DIFFER = Sensitivity.ONE_DIFFER


def generate_arguments(algorithm, d1, d2, default_kwargs):
    """
    :param algorithm: The algorithm to test for.
    :param d1: The database 1
    :param d2: The database 2
    :param default_kwargs: The default arguments that are given or have a default value.
    :return: Extra argument needed for the algorithm besides Q and epsilon.
    """
    arguments = algorithm.__code__.co_varnames[:algorithm.__code__.co_argcount]
    if arguments[2] not in default_kwargs:
        logger.error(f'The third argument {arguments[2]} (privacy budget) is not provided!')
        return None

    return default_kwargs


def generate_databases(algorithm, num_input, default_kwargs, sensitivity=ALL_DIFFER):

    if not isinstance(sensitivity, Sensitivity):
        raise ValueError('sensitivity must be statdp.ALL_DIFFER or statdp.ONE_DIFFER')

    # assume maximum distance is 1
    d1 = [1 for _ in range(num_input)]
    candidates = [
        (d1, [0] + [1 for _ in range(num_input - 1)]),  # one below
        (d1, [2] + [1 for _ in range(num_input - 1)]),  # one above
    ]

    if sensitivity == ALL_DIFFER:
        candidates.extend([
            (d1, [2] + [0 for _ in range(num_input - 1)]),  # one above rest below
            (d1, [0] + [2 for _ in range(num_input - 1)]),  # one below rest above
            # half half
            (d1, [2 for _ in range(int(num_input / 2))] + [0 for _ in range(num_input - int(num_input / 2))]),
            (d1, [2 for _ in range(num_input)]),  # all above
            (d1, [0 for _ in range(num_input)]),  # all below
            # x shape
            ([1 for _ in range(int(math.floor(num_input / 2.0)))] + [0 for _ in range(int(math.ceil(num_input / 2.0)))],
             [0 for _ in range(int(math.floor(num_input / 2.0)))] + [1 for _ in range(int(math.ceil(num_input / 2.0)))])
        ])

    return tuple((d1, d2, generate_arguments(algorithm, d1, d2, default_kwargs)) for d1, d2 in candidates)



#Selectors 

def _evaluate_input(input_triplet, algorithm, iterations):
    d1, d2, kwargs = input_triplet
    return run_algorithm(algorithm, d1, d2, kwargs, None, iterations)


def select_event(algorithm, input_list, epsilon, iterations, process_pool, quiet=False):
    
    if not callable(algorithm):
        raise ValueError('Algorithm must be callable')

    # fill in other arguments for _evaluate_input function, leaving out `input` to be filled
    partial_evaluate_input = functools.partial(_evaluate_input, algorithm=algorithm, iterations=iterations)

    threshold = 0.001 * iterations * np.exp(epsilon)
    
    event_evaluator = tqdm.tqdm(process_pool.imap_unordered(partial_evaluate_input, input_list),
                                desc='Finding best inputs/events', total=len(input_list), unit='input', leave=False,
                                disable=quiet)
    # flatten the results for all input/event pairs
    counts, input_event_pairs, p_values = [], [], []
    for local_counts, local_input_event_pair in event_evaluator:
        # put the results in the list for later references
        counts.extend(local_counts)
        input_event_pairs.extend(local_input_event_pair)

        # calculate p-values based on counts
        for (cx, cy) in local_counts:
            p_values.append(test_statistics(cx, cy, epsilon, iterations) if cx + cy > threshold else float('inf'))

    # log the information for debug purposes
    for ((d1, d2, kwargs, event), (cx, cy), p) in zip(input_event_pairs, counts, p_values):
        logger.debug(f"d1: {d1} | d2: {d2} | kwargs: {kwargs} | event: {event} | p-value: {p:5.3f} | "
                     f"cx: {cx} | cy: {cy} | ratio: {float(cy) / cx if cx != 0 else float('inf'):5.3f}")

    # find an (d1, d2, kwargs, event) pair which has minimum p value from search space
    return input_event_pairs[np.asarray(p_values).argmin()]






# Hypothesis Testing 




@numba.njit
def test_statistics(cx, cy, epsilon, iterations):
    """ Calculate p-value based on observed results.
    :param cx: The observed count of running algorithm with database 1 that falls into the event
    :param cy:The observed count of running algorithm with database 2 that falls into the event
    :param epsilon: The epsilon to test for.
    :param iterations: The total iterations for running algorithm.
    :return: p-value
    """
    # average p value
    sample_num = 200
    p_value = 0
    for new_cx in np.random.binomial(cx, 1.0 / (np.exp(epsilon)), sample_num):
        p_value += sf(new_cx - 1, 2 * iterations, iterations, new_cx + cy)
    return p_value / sample_num


def hypothesis_test(algorithm, d1, d2, kwargs, event, epsilon, iterations, process_pool, report_p2=True):
    
    # use undocumented mp.Pool._processes to get the number of max processes for the pool, this is unstable and
    # may break in the future, therefore we fall back to mp.cpu_count() if it is not accessible
    core_count = process_pool._processes if process_pool._processes and isinstance(process_pool._processes, int) \
        else mp.cpu_count()
    if iterations < core_count:
        process_iterations = [iterations]
    else:
        process_iterations = [int(math.floor(float(iterations) / core_count)) for _ in range(core_count)]
        # add the remaining iterations to the last index
        process_iterations[core_count - 1] += iterations % process_iterations[core_count - 1]

    # start the pool to run the algorithm and collects the statistics
    cx, cy = 0, 0
    # fill in other arguments for running the algorithm, leaving `iterations` to be filled
    runner = functools.partial(run_algorithm, algorithm, d1, d2, kwargs, event)
    for ((local_cx, local_cy), *_), _ in process_pool.imap_unordered(runner, process_iterations):
        cx += local_cx
        cy += local_cy
    cx, cy = (cx, cy) if cx > cy else (cy, cx)

    # calculate and return p value
    if report_p2:
        return test_statistics(cx, cy, epsilon, iterations), test_statistics(cy, cx, epsilon, iterations)
    else:
        return test_statistics(cx, cy, epsilon, iterations)




#Run Algorithms 



def run_algorithm(algorithm, d1, d2, kwargs, event, total_iterations):
    """ Run the algorithm for :iteration: times, count and return the number of iterations in :event:,
    event search space is auto-generated if not specified.
    :param algorithm: The algorithm to run.
    :param d1: The D1 input to run.
    :param d2: The D2 input to run.
    :param kwargs: The keyword arguments for the algorithm.
    :param event: The event to test, auto generate event search space if None.
    :param total_iterations: The iterations to run.
    :return: [(cx, cy), ...], [(d1, d2, kwargs, event), ...]
    """
    if not callable(algorithm):
        raise ValueError('Algorithm must be callable')
    prng = np.random.default_rng()
    # support multiple return values, each return value is stored as a row in result_d1 / result_d2
    # e.g if an algorithm returns (1, 1), result_d1 / result_d2 would be like
    # [
    #   [x, x, x, ..., x],
    #   [x, x, x, ..., x]
    # ]

    # get return type by a sample run
    all_possible_events = None
    event_dict = {}
    sample_result = algorithm(prng, d1, **kwargs)

    # since we need to store the output in intermediate variables (`result_d1` and `result_d2`), if the total
    # iterations are very large, peak memory usage would kill the program, therefore we divide the
    if total_iterations > int(1e6):
        logger.debug('Iterations too large, divide into different pieces')
        iteration_tuple = [int(1e6) for _ in range(math.floor(total_iterations / 1e6))] + [total_iterations % int(1e6)]
    else:
        iteration_tuple = (total_iterations,)
    for iterations in iteration_tuple:
        if np.issubdtype(type(sample_result), np.number):
            result_d1 = (np.fromiter((algorithm(prng, d1, **kwargs) for _ in range(iterations)),
                                     dtype=type(sample_result), count=iterations),)
            result_d2 = (np.fromiter((algorithm(prng, d2, **kwargs) for _ in range(iterations)),
                                     dtype=type(sample_result), count=iterations),)
        elif isinstance(sample_result, (tuple, list)):
            # create a list of numpy array, each containing the output from running
            result_d1, result_d2 = [np.empty(iterations, dtype=type(sample_result[result_index])) for result_index in
                                    range(len(sample_result))], \
                                   [np.empty(iterations, dtype=type(sample_result[result_index])) for result_index in
                                    range(len(sample_result))],

            for iteration_number in range(iterations):
                out_1 = algorithm(prng, d1, **kwargs)
                out_2 = algorithm(prng, d2, **kwargs)
                for row, (value_1, value_2) in enumerate(zip(out_1, out_2)):
                    result_d1[row][iteration_number] = value_1
                    result_d2[row][iteration_number] = value_2
        else:
            raise ValueError(f'Unsupported return type: {type(sample_result)}')

        # if possible events are not determined yet
        if not all_possible_events:
            # get desired search space for each return value
            event_search_space = []
            if event is None:
                for row in range(len(result_d1)):
                    # determine the event search space based on the return type
                    combined_result = np.concatenate((result_d1[row], result_d2[row]))
                    unique = np.unique(combined_result)

                    # categorical output
                    if len(unique) < iterations * 0.002:
                        event_search_space.append(tuple(int(key) for key in unique))
                    else:
                        combined_result.sort()
                        # find the densest 70% range
                        search_range = int(0.7 * len(combined_result))
                        search_max = min(range(search_range, len(combined_result)),
                                         key=lambda x: combined_result[x] - combined_result[x - search_range])
                        search_min = search_max - search_range

                        event_search_space.append(
                            tuple((-float('inf'), float(alpha)) for alpha in
                                  np.linspace(combined_result[search_min], combined_result[search_max], num=10)))

                logger.debug(f"search space is set to {' × '.join(str(event) for event in event_search_space)}")
            else:
                # if `event` is given, it should have the corresponding events for each return value
                if len(event) != len(result_d1):
                    raise ValueError('Given event should have the same dimension as return value.')
                # here if the event is given, we carefully construct the search space in the following format:
                # [first_event] × [second_event] × [third_event] × ... × [last_event]
                # so that when the search begins, only one possible combination can happen which is the given event
                event_search_space = ((separate_event,) for separate_event in event)
            all_possible_events = tuple(itertools.product(*event_search_space))

        for event in all_possible_events:
            cx_check, cy_check = np.full(iterations, True, dtype=np.bool_), np.full(iterations, True, dtype=np.bool_)
            # check for all events in the return values
            for row in range(len(result_d1)):
                if np.issubdtype(type(event[row]), np.number):
                    cx_check = np.logical_and(cx_check, result_d1[row] == event[row])
                    cy_check = np.logical_and(cy_check, result_d2[row] == event[row])
                else:
                    cx_check = np.logical_and(cx_check, np.logical_and(result_d1[row] > event[row][0],
                                                                       result_d1[row] < event[row][1]))
                    cy_check = np.logical_and(cy_check, np.logical_and(result_d2[row] > event[row][0],
                                                                       result_d2[row] < event[row][1]))

            cx, cy = np.count_nonzero(cx_check), np.count_nonzero(cy_check)
            if event not in event_dict:
                event_dict[event] = (cx, cy)
            else:
                old_cx, old_cy = event_dict[event]
                event_dict[event] = cx + old_cx, cy + old_cy

    counts, input_event_pairs = [], []
    for event, (cx, cy) in event_dict.items():
        counts.append((cx, cy) if cx > cy else (cy, cx))
        input_event_pairs.append((d1, d2, kwargs, event))
    return counts, input_event_pairs



#Binomial 


@numba.njit(numba.float64(numba.int_, numba.int_))
def _ln_binomial(n, k):
    """log of binomial coefficient function (n k), i.e., n choose k"""
    if k > n:
        raise ValueError
    if k == n or k == 0:
        return 0
    if k * 2 > n:
        k = n - k
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


@numba.njit(numba.float64(numba.int_, numba.int_, numba.int_, numba.int_))
def pmf(k, M, n, N):
    """returns the pmf of hypergeometric distribution for given parameters. This interface mimics scipy's hypergeom.pmf
    :param k: input value
    :param M: the total number of objects
    :param n: the total number of Type 1 objects
    :param N: the number of draws
    :return: the probability mass function for given parameter
    """
    if N > M:
        raise ValueError
    if k > n or k > N:
        return 0
    elif N > M - n and k + M - n < N:
        return 0
    return math.exp(_ln_binomial(n, k) + _ln_binomial(M - n, N - k) - _ln_binomial(M, N))


@numba.njit(numba.float64(numba.int_, numba.int_, numba.int_, numba.int_))
def sf(k, M, n, N):
    """returns the survival function of hypergeometric distribution for given parameters. This equals (1 - cdf) but we
    try to be more precise than (1 - cdf). This interface mimics scipy.stats.hypergeom.sf.
    :param k: input value
    :param M: the total number of objects
    :param n: the total number of Type 1 objects
    :param N: the number of draws
    :return: the cumulative density for given parameter
    """
    if N > M:
        raise ValueError('The number of draws (N) is larger than the total number of objects (M)')
    if k >= min(n, N):
        return 0
    elif k < 0:
        return 1
    # calculating the pmf is expensive, use the following recursive definition for performance:
    # P(X=i) = (i / (n - i + 1)) * ((M - n + i - N) / (N - i + 1)) * P(X=i+1)
    # P(X=i) = ((n - i) / (i + 1)) * ((N - i) / (M - n + i + 1 - N)) * P(X=i-1)

    # the hypergeometric distribution is centered around N * (n / M),
    # i.e. pmf has the largest value when k = N * (n / M)
    # therefore, for fewer iterations, we use forward recursive definition to calculate P(X > k) for k > N * (n / M)
    # otherwise we use backward recursive definition to calculate P(X <= k) and return 1 - P(x <= k)
    # this also gives use more precise result when pmf(k) ~= 0 since the error will be significant and propagated
    # through the recursion
    if k > N * n / M:
        pmf_i = pmf(k + 1, M, n, N)
        result = pmf_i
        for i in range(k + 1, N):
            pmf_i *= ((n - i) / (i + 1)) * ((N - i) / (M - n + i + 1 - N))
            result += pmf_i
        return result
    else:
        pmf_i = pmf(k, M, n, N)
        result = pmf_i
        for i in range(k, 0, -1):
            pmf_i *= (i / (n - i + 1)) * ((M - n + i - N) / (N - i + 1))
            result += pmf_i
        return 1 - result




# Integration
def detect_counterexample(algorithm, test_epsilon, default_kwargs=None, databases=None, num_input=(5, 10),
                          event_iterations=1000, detect_iterations=5000, cores=None, sensitivity=ALL_DIFFER,
                          quiet=False, loglevel=logging.INFO):
    """
    :param algorithm: The algorithm to test for.
    :param test_epsilon: The privacy budget to test for, can either be a number or a tuple/list.
    :param default_kwargs: The default arguments the algorithm needs except the first Queries argument.
    :param databases: The databases to run for detection, optional.
    :param num_input: The length of input to generate, not used if database param is specified.
    :param event_iterations: The iterations for event selector to run.
    :param detect_iterations: The iterations for detector to run.
    :param cores: The number of max processes to set for multiprocessing.Pool(), os.cpu_count() is used if None.
    :param sensitivity: The sensitivity setting, all queries can differ by one or just one query can differ by one.
    :param quiet: Do not print progress bar or messages, logs are not affected.
    :param loglevel: The loglevel for logging package.
    :return: [(epsilon, p, d1, d2, kwargs, event)] The epsilon-p pairs along with databases/arguments/selected event.
    """
    # initialize an empty default kwargs if None is given
    default_kwargs = default_kwargs if default_kwargs else {}

    logging.basicConfig(level=loglevel)
    logger.info(f'Start detection for counterexample on {algorithm.__name__} with test epsilon {test_epsilon}')
    logger.info(f'Options -> default_kwargs: {default_kwargs} | databases: {databases} | cores:{cores}')
    
    input_list = []
    if databases is not None:
        d1, d2 = databases
        kwargs = generate_arguments(algorithm, d1, d2, default_kwargs=default_kwargs)
        input_list = ((d1, d2, kwargs),)
    else:
        num_input = (int(num_input), ) if isinstance(num_input, (int, float)) else num_input
        for num in num_input:
            input_list.extend(
                generate_databases(algorithm, num, default_kwargs=default_kwargs, sensitivity=sensitivity))

    result = []
    # convert int/float or iterable into tuple (so that it has length information)
    test_epsilon = (test_epsilon, ) if isinstance(test_epsilon, (int, float)) else test_epsilon

    with mp.Pool(cores) as pool:
        for _, epsilon in tqdm.tqdm(enumerate(test_epsilon), total=len(test_epsilon), unit='test', desc='Detection',
                                    disable=quiet):
            d1, d2, kwargs, event = select_event(algorithm, input_list, epsilon, event_iterations, quiet=quiet,
                                                 process_pool=pool)
           
            p = hypothesis_test(algorithm, d1, d2, kwargs, event, epsilon, detect_iterations, report_p2=False,
                                process_pool=pool)
            result.append((epsilon, float(p), d1, d2, kwargs, event))
            if not quiet:
                tqdm.tqdm.write(f'Epsilon: {epsilon} | p-value: {p:5.3f} | Event: {event}')
            logger.debug(f'D1: {d1} | D2: {d2} | kwargs: {kwargs}')
       
        
        return result








# Plotting Results

def plot_result(data, xlabel, ylabel, title, output_filename):
    """plot the results similar to the figures in our paper
    :param data: The input data sets to plots. e.g., {algorithm_epsilon: [(test_epsilon, pvalue), ...]}
    :param xlabel: The label for x axis.
    :param ylabel: The label for y axis.
    :param title: The title of the figure.
    :param output_filename: The output file name.
    :return: None
    """
    # setup the figure
    plt.ylim(0.0, 1.0)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    # colors and markers for each claimed epsilon
    markers = ['s', 'o', '^', 'x', '*', '+', 'p']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    # add an auxiliary line for p-value=0.05
    plt.axhline(y=0.05, color='black', linestyle='dashed', linewidth=1.2)
    for i, (epsilon, points) in enumerate(data.items()):
        # add an auxiliary vertical line for the claimed privacy
        plt.axvline(x=float(epsilon), color=colors[i % len(colors)], linestyle='dashed', linewidth=1.2)
        # plot the
        x = [item[0] for item in points]
        p = [item[1] for item in points]
        plt.plot(x, p, 'o-',
                 label=f'$\\epsilon_0$ = {epsilon}', markersize=8, marker=markers[i % len(markers)], linewidth=3)

    # plot legends
    legend = plt.legend()
    legend.get_frame().set_linewidth(0.0)

    # save the figure and clear the canvas for next draw
    plt.savefig(output_filename, bbox_inches='tight')
    plt.gcf().clear()


def main():
    print("Helloauaubfiuabsbfasdnkanskjdnjaksndjka")
    print("Helloauaubfiuabsbfasdnkanskjdnjaksndjka")
    print("Helloauaubfiuabsbfasdnkanskjdnjaksndjka")
    # list of tasks to test, each tuple contains (function, extra_args, sensitivity)
    tasks = [
        (noisy_max_v1a, {}, ALL_DIFFER),
        (noisy_max_v1b, {}, ALL_DIFFER),
        (noisy_max_v2a, {}, ALL_DIFFER),
        (noisy_max_v2b, {}, ALL_DIFFER),
        (histogram, {}, ONE_DIFFER),
        (histogram_eps, {}, ONE_DIFFER),
        (SVT, {'N': 1, 'T': 0.5}, ALL_DIFFER),
    ]

    # claimed privacy level to check
    claimed_privacy = (0.2, 0.7, 1.5)

    # privacy levels to test, here we test from a range of 0.1 - 2.0 with a stepping of 0.1
    test_privacy = tuple(x / 10.0 for x in range(1, 20, 1))

    for i, (algorithm, kwargs, sensitivity) in enumerate(tasks):
        start_time = time.time()
        results = {}
        for privacy_budget in claimed_privacy:
            # set the third argument of the function (assumed to be `epsilon`) to the claimed privacy level
            kwargs[algorithm.__code__.co_varnames[2]] = privacy_budget
            print(kwargs)
            
            results[privacy_budget] = detect_counterexample(algorithm, test_privacy, kwargs, sensitivity=sensitivity)
            print("Helloauaubfiuabsbfasdnkanskjdnjaksndjka")

        # dump the results to file
        
        json_file = pathlib.Path.cwd() / f'{algorithm.__name__}.json'
        if json_file.exists():
            logger.warning(f'{algorithm.__name__}.json already exists, note that it will be over-written')
            json_file.unlink()

        with json_file.open('w') as f:
            json.dump(results, f)

        # plot and save to file
        plot_file = pathlib.Path.cwd() / f'{algorithm.__name__}.pdf'
        if plot_file.exists():
            logger.warning(f'{algorithm.__name__}.pdf already exists, it will be over-written')
            plot_file.unlink()

        plot_result(results, r'Test $\epsilon$', 'P Value', algorithm.__name__.replace('_', ' ').title(), plot_file)

        total_time, total_detections = time.time() - start_time, len(claimed_privacy) * len(test_privacy)
        logger.info(f'[{i + 1} / {len(tasks)}]: {algorithm.__name__} | Time elapsed: {total_time:5.3f}s | '
                    f'Average time per detection: {total_time / total_detections:5.3f}s')


if __name__ == '__main__':
    main()