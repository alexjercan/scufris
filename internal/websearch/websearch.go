package websearch

import (
	"context"
	"fmt"
	"net/http"
	"net/url"

	"github.com/PuerkitoBio/goquery"
	"github.com/alexjercan/scufris"
)

// DefaultUserAgent defines a default value for user-agent header.
const DefaultUserAgent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36"

// Result holds the returned query data
type Result struct {
	Title string
	Info  string
	URL   string
}

type WebSearchClient interface {
	Search(ctx context.Context, query string, maxResult int) ([]Result, error)
}

type DdgClient struct {
	httpClient *http.Client
}

func NewDdgClient() WebSearchClient {
	return &DdgClient{
		httpClient: http.DefaultClient,
	}
}

func (c *DdgClient) Search(ctx context.Context, query string, maxResult int) ([]Result, error) {
	results := []Result{}
	queryUrl := fmt.Sprintf("https://html.duckduckgo.com/html/?q=%s", url.QueryEscape(query))

	req, err := http.NewRequestWithContext(ctx, "GET", queryUrl, nil)
	if err != nil {
		return results, &scufris.Error{
			Code:    "DDG_REQUEST_ERROR",
			Message: "failed to create request",
			Err:     fmt.Errorf("failed to create request: %w", err),
		}
	}
	req.Header.Set("User-Agent", DefaultUserAgent)

	res, err := c.httpClient.Do(req)
	if err != nil {
		return nil, &scufris.Error{
			Code:    "DDG_REQUEST_ERROR",
			Message: "failed to make request",
			Err:     fmt.Errorf("failed to make request: %w", err),
		}
	}

	doc, err := goquery.NewDocumentFromReader(res.Body)
	if err != nil {
		return nil, &scufris.Error{
			Code:    "DDG_REQUEST_ERROR",
			Message: "failed to parse response",
			Err:     fmt.Errorf("failed to parse response: %w", err),
		}
	}

	sel := doc.Find(".web-result")

	for i := range sel.Nodes {
		// Break loop once required amount of results are add
		if maxResult == len(results) {
			break
		}
		node := sel.Eq(i)
		titleNode := node.Find(".result__a")

		info := node.Find(".result__snippet").Text()
		title := titleNode.Text()
		ref := ""

		if len(titleNode.Nodes) > 0 && len(titleNode.Nodes[0].Attr) > 2 {
			ref = titleNode.Nodes[0].Attr[2].Val
		}

		results = append(results[:], Result{title, info, ref})

	}

	return results, nil
}
