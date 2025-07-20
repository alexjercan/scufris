package registry

import (
	"github.com/google/uuid"
	"github.com/uptrace/bun"
)

type Image struct {
    bun.BaseModel `bun:"table:images,alias:i"`

	ID	 uuid.UUID `bun:"id,pk,type:uuid"`
	Blob string `bun:"blob,type:text,notnull"`
}
