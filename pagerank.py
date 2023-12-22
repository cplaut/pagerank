import os
import random
import re
import sys
import numpy as np


DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    keys = list(corpus.keys())
    pages = corpus[page]

    probdis = dict()

    for key in keys:
        linked_bonus = 0
        if key in pages:
            linked_bonus = damping_factor/len(pages)
        if len(pages) == 0:
            linked_bonus = damping_factor/len(keys)
        probdis[key] = (1-damping_factor)/len(keys) + linked_bonus

    return probdis


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    pageranks = dict()
    pageranks_raw = []
    cur_page = random.choice(list(corpus.keys()))
    pageranks_raw.append(cur_page)
    for _ in range(n-1):
        prob_dis = transition_model(corpus, cur_page, damping_factor)
        probs = []

        for key in prob_dis.keys():
            probs.append(prob_dis[key])

        cur_page = random.choices(list(prob_dis.keys()), weights = probs, k=1).pop()
        pageranks_raw.append(cur_page)

    for key in corpus.keys():
        pageranks[key] = pageranks_raw.count(key)/n

    return pageranks


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    pages = list(corpus.keys())
    prob_dis = dict()
    
    for page in pages:
        prob_dis[page] = 1/len(pages)

    converged = False

    while not(converged):
        previous_prob_dis = dict(prob_dis)
        prob_dis_diff = np.array(list())
        for page in pages:
            partial_prob = 0
            for parent in set(pages).difference(set([page])):
                if page in corpus[parent]:
                    partial_prob += previous_prob_dis[parent]/len(corpus[parent])
                elif len(corpus[parent]) == 0:
                    partial_prob += 1/len(pages)

            prob_dis[page] = (1-damping_factor)/len(pages) + (damping_factor * partial_prob)

            prob_dis_diff = np.append(prob_dis_diff, [abs(previous_prob_dis[page] - prob_dis[page])])

        if not((prob_dis_diff>0.001).any()):
            converged = True

    return prob_dis
            


if __name__ == "__main__":
    main()