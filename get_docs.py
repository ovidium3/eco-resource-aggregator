import base64
import io
import json
import os
import random
import re
import requests
import time

from contextlib import suppress
from dotenv import load_dotenv


####################################################################################################
# GLOBALS
####################################################################################################

load_dotenv()
API_KEY = os.getenv("CORE_API_KEY")     # from .env to make API calls to CORE
OUTPUT_DIR = "climate_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)  # output directory containing 13 JSON files, 1 for each category

TARGET_PAPERS = 10          # target number of papers containing full JSON components
BATCH_SIZE = 25             # number of papers searched per batch
MAX_RETRIES = 5             # max number of retries for API requests on error
RETRY_DELAY_BASE = 3.0      # base delay for exponential backoff when encountering errors


def is_valid_text(text):
    """
    Checks if the downloaded text is actually readable text, and not PDF markup or binary data.
    Args:
        text (str): The text to check.
    Returns:
        bool: True if the text is valid, False otherwise."""
    if not text:
        return False
        
    # check if it is a PDF file
    if text.startswith("%PDF-"):
        print("  ✗ Received PDF data instead of text")
        return False
        
    # check if it is binary data
    if "\0" in text or text.count('\ufffd') > 5:
        print("  ✗ Received binary data")
        return False
        
    # check for sufficient alphanumeric content
    alphanumeric_count = sum(c.isalnum() for c in text)
    if alphanumeric_count < 200:
        print(f"  ✗ Insufficient content: only {alphanumeric_count} alphanumeric chars")
        return False
        
    # check for a reasonable word count
    words = re.findall(r'\b\w+\b', text)
    if len(words) < 100:
        print(f"  ✗ Insufficient content: only {len(words)} words")
        return False
        
    return True


def fetch_with_retry(func, *args, **kwargs):
    """
    General retry function with exponential backoff in case of errors.
    Args:
        func (callable): The function to call.
        *args: Positional arguments for the function.
        **kwargs: Keyword arguments for the function.
    Returns:
        The result of the function call, or None if all retries fail.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return func(*args, **kwargs)
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:   # not found err: caused by a missing resource
                print(f"  ✗ Resource not found: {e}")
                return None
            elif e.response.status_code == 429:     # rate limit error: too many requests
                wait = int(e.response.headers.get('X-RateLimit-Retry-After', RETRY_DELAY_BASE * 2**attempt))
                print(f"  Rate limit hit. Waiting {wait}s before retry.")
                time.sleep(wait)
            else:       # other HTTP errors, perhaps server errors
                wait = RETRY_DELAY_BASE * (2 ** (attempt - 1)) + random.random()
                print(f"  Retry {attempt}/{MAX_RETRIES} in {wait:.1f}s -> {e}")
                time.sleep(wait)
        except Exception as e:  # other exceptions, e.g., connection errors
            wait = RETRY_DELAY_BASE * (2 ** (attempt - 1)) + random.random()
            print(f"  Retry {attempt}/{MAX_RETRIES} in {wait:.1f}s -> {e}")
            time.sleep(wait)
            
        if attempt == MAX_RETRIES:  # if we reach the max retries (5), log the error and halt
            print(f"  Giving up after {MAX_RETRIES} retries")
            return None
            
    return None


def search_papers(query, page=1, page_size=BATCH_SIZE):
    """
    Searches for papers, focusing on those likely to have text.
    Args:
        query (str): The search query.
        page (int): The page number to fetch.
        page_size (int): The number of results per page.
    Returns:
        list: A list of papers matching the query.
    """
    url = "https://api.core.ac.uk/v3/search/works"  # search through works as opposed to authors or just abstracts
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    # use detailed query to filter for papers more likely to have text
    params = {
        "q": f"{query}",        # Use targeted query --> category
        "offset": (page - 1) * page_size,
        "limit": page_size,
        # Request more fields to have multiple options for content in case fundamental 4 fields are empty
        "fields": "id,doi,title,abstract,fullText,downloadUrl,publisher,language"
    }
    
    # make the API request with proper authentication and parameters to extract files while minimizing errors
    print(f"  Searching page {page} for '{query}'")
    try:
        response = requests.get(url, headers=headers, params=params, timeout=60)
        
        if 'X-RateLimit-Remaining' in response.headers:
            print(f"  Rate limit remaining: {response.headers.get('X-RateLimit-Remaining')}")
            
        response.raise_for_status()
        results = response.json().get("results", [])
        print(f"  Found {len(results)} results on page {page}")
        return results
    except Exception as e:
        print(f"  Search error: {e}")
        raise


def get_detailed_metadata(work_id):
    """
    Gets detailed metadata which might include more text.
    Args:
        work_id (str): The ID of the work.
    Returns:
        dict: The detailed metadata of the work.
    """
    url = f"https://api.core.ac.uk/v3/works/{work_id}"
    headers = {"Authorization": f"Bearer {API_KEY}"}
    
    # use the work ID to get detailed metadata
    print(f"  Getting detailed metadata for work ID {work_id}")
    try:
        response = requests.get(url, headers=headers, timeout=60)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"  Error getting detailed metadata: {e}")
        return None


def try_different_download_methods(work_id):
    """
    Tries multiple methods to get text content.
    Args:
        work_id (str): The ID of the work.
    Returns:
        str: The text content if successful, None otherwise.
    """
    
    # Method 1: try direct full text download
    try:
        url = f"https://api.core.ac.uk/v3/works/{work_id}/download"
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Accept": "text/plain"
        }
        
        print(f"  Trying direct text download for work ID {work_id}")
        response = requests.get(url, headers=headers, timeout=90)
        response.raise_for_status()
        
        text = response.text
        if is_valid_text(text):
            print(f"  ✓ Method 1 success: {len(text)} chars of text")
            return text
    except Exception as e:
        print(f"  Method 1 failed: {e}")
    
    # Method 2: try to get from detailed metadata
    try:
        metadata = get_detailed_metadata(work_id)
        if metadata and 'fullText' in metadata and metadata['fullText']:
            text = metadata['fullText']
            if is_valid_text(text):
                print(f"  ✓ Method 2 success: {len(text)} chars of text")
                return text
    except Exception as e:
        print(f"  Method 2 failed: {e}")
    
    # Method 3: try different download URL format
    try:
        url = f"https://api.core.ac.uk/v3/download/{work_id}"
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Accept": "text/plain"
        }
        
        print(f"  Trying alternate download URL format")
        response = requests.get(url, headers=headers, timeout=90)
        response.raise_for_status()
        
        text = response.text
        if is_valid_text(text):
            print(f"  ✓ Method 3 success: {len(text)} chars of text")
            return text
    except Exception as e:
        print(f"  Method 3 failed: {e}")
    
    print("All text download methods failed.")
    return None


def collect_papers_with_text(query, target_count=TARGET_PAPERS):
    """
    Collects papers with verified text content.
    Args:
        query (str): The search query.
        target_count (int): The target number of papers to collect.
    Returns:
        list: A list of papers with text content.
    """
    papers_with_text = []
    page = 1
    max_pages = 100
    
    # load previous progress if exists
    temp_file = os.path.join(OUTPUT_DIR, f"{query.replace(' ', '_')}_temp.json")
    if os.path.exists(temp_file):
        try:
            with open(temp_file, 'r', encoding='utf-8') as f:
                papers_with_text = json.load(f)
                print(f"  Loaded {len(papers_with_text)} papers from previous run")
        except Exception as e:
            print(f"Error loading previous progress: {e}")
    
    # resume where we left off
    if papers_with_text:
        page = (len(papers_with_text) // BATCH_SIZE) + 1
        print(f"Resuming from page {page}")
    
    # keep searching until we have enough papers with text in them
    while len(papers_with_text) < target_count and page <= max_pages:
        try:
            batch = fetch_with_retry(search_papers, query, page, BATCH_SIZE)
            
            if not batch:
                print(f"No results on page {page} or search error")
                page += 1
                time.sleep(5)
                continue
            
            # process each paper in batch
            for paper in batch:
                work_id = paper.get('id')
                if not work_id:
                    continue
                
                # skip papers we alr have
                if any(p.get('id') == work_id for p in papers_with_text):
                    print(f"Already have paper ID {work_id}, skipping")
                    continue
                
                print(f"Processing: {paper.get('title', 'Untitled')[:50]}... (ID: {work_id})")
                
                # check if paper already has fullText in the search results
                if 'fullText' in paper and paper['fullText'] and is_valid_text(paper['fullText']):
                    full_text = paper['fullText']
                    print(f"Paper already has full text: {len(full_text)} chars")
                else:
                    # try different methods to get text
                    full_text = try_different_download_methods(work_id)
                
                if full_text:
                    # create paper record with text
                    paper_with_text = {
                        'id': work_id,
                        'doi': paper.get('doi'),
                        'title': paper.get('title'),
                        'abstract': paper.get('abstract'),
                        'fullText': full_text,
                        'source': paper.get('publisher', 'Unknown')
                    }
                    
                    papers_with_text.append(paper_with_text)
                    print(f"Success! Papers with text: {len(papers_with_text)}/{target_count}")
                    
                    # save progress after each successful paper
                    with open(temp_file, 'w', encoding='utf-8') as f:
                        json.dump(papers_with_text, f)
                    
                    # check if we reached our target
                    if len(papers_with_text) >= target_count:
                        print(f"Target reached: {target_count} papers with text")
                        break
                
                time.sleep(3)   # give the API a break
            
            page += 1
            print(f"Moving to page {page}...")
            time.sleep(5)   # give the API another break
            
        except Exception as e:
            print(f"Error on page {page}: {e}")
            time.sleep(15)
            page += 1
    
    # clean up temp file when done
    if os.path.exists(temp_file) and len(papers_with_text) >= target_count:
        try:
            os.remove(temp_file)
            print("Cleaned up temporary file")
        except:
            pass
    
    return papers_with_text[:target_count]


def main():
    """
    Main function which extracts keywords from Climate-Change-NER, and downloads research papers from CORE API.
    """
    # same as anchors in prune_climate_kws.py, obtained from Climate-Change-NER predefined categories
    categories = [
        "climate assets", "climate datasets", "greenhouse gases", "climate hazards",
        "climate impacts", "climate mitigation", "climate models", "climate nature",
        "climate observations", "climate organisms", "climate organizations",
        "origins of climate problems", "climate properties",
    ]

    print("Downloading research papers from CORE API")

    # Loop through each category and download papers
    # from the CORE API, saving them to separate JSON files in output dir
    for category in categories:
        output_file = os.path.join(OUTPUT_DIR, f"{category.lower().replace(' ', '_')}.json")
        
        if os.path.exists(output_file):
            print(f"\nSkipping '{category}' (already completed)")
            continue
            
        print(f"\nCategory: {category}")
        
        try:
            papers = collect_papers_with_text(category, TARGET_PAPERS)
            
            if papers:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(papers, f, indent=2)
                print(f"Saved {len(papers)} papers to {output_file}")
            else:
                print(f"No papers with text found for '{category}'")
                
        except Exception as e:
            print(f"Error processing '{category}': {e}")
            
        print(f"Waiting before next category...")
        time.sleep(20)  # sleep to avoid hitting API rate limits or timeout

    print("\nAll done!")


if __name__ == "__main__":
    main()
