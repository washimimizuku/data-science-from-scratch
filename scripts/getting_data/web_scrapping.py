from bs4 import BeautifulSoup
import requests

# I put the relevant HTML file on GitHub. In order to fit
# the URL in the book I had to split it across two lines.
# Recall that whitespace-separated strings get concatenated.
url = ("https://raw.githubusercontent.com/"
       "joelgrus/data/master/getting-data.html")
html = requests.get(url).text
soup = BeautifulSoup(html, 'html5lib')

first_paragraph = soup.find('p') # Or just soup.p

print('first_paragraph:', first_paragraph)

assert str(first_paragraph) == '<p id="p1">This is the first paragraph.</p>'

first_paragraph_text = soup.p.text
first_paragraph_words = soup.p.text.split()

print('first_paragraph_text:', first_paragraph_text)
print('first_paragraph_words:', first_paragraph_words)

assert first_paragraph_words == ['This', 'is', 'the', 'first', 'paragraph.']

first_paragraph_id = soup.p['id'] # Raises KeyError if no 'id'
first_paragraph_id2 = soup.p.get('id') # Returns None if no 'id'

print('first_paragraph_id:', first_paragraph_id)
print('first_paragraph_id2:', first_paragraph_id2)

assert first_paragraph_id == first_paragraph_id2 == 'p1'

all_paragraphs = soup.find_all('p') # Or just soup('p')
paragraphs_with_ids = [p for p in soup('p') if p.get('id')]

print('all_paragraphs:', all_paragraphs)
print('paragraphs_with_ids:', paragraphs_with_ids)

assert len(all_paragraphs) == 2
assert len(paragraphs_with_ids) == 1

important_paragraphs = soup('p', {'class': 'important'})
important_paragraphs2 = soup('p', 'important')
important_paragraphs3 = [p for p in soup('p')
                        if 'important' in p.get('class', [])]

print('important_paragraphs:', important_paragraphs)
print('important_paragraphs2:', important_paragraphs3)
print('important_paragraphs3:', important_paragraphs3)

assert important_paragraphs == important_paragraphs2 == important_paragraphs3
assert len(important_paragraphs) == 1

# Warning, will return the same span multiple times
# if it sits inside multiple divs.
# be more clever if that's the case.
spans_inside_divs = [span
                     for div in soup('div') # For each <div> on the page
                     for span in div('span')] # Find each <span> inside it

print('spans_inside_divs:', spans_inside_divs)

assert len(spans_inside_divs) == 3