package knowledge

import (
	"github.com/google/uuid"
	"github.com/uptrace/bun"
)

type KnowledgeSource struct {
	bun.BaseModel `bun:"table:knowledge_sources,alias:ks"`

	ID          uuid.UUID `bun:"id,pk,type:uuid"`
	Name        string    `bun:"name,notnull"`
	Description string    `bun:"description,notnull"`

	Knowledges []Knowledge `bun:"rel:has-many,join:id=source_id"`
}

func NewKnowledgeSource(name, description string) *KnowledgeSource {
	return &KnowledgeSource{
		ID:          uuid.New(),
		Name:        name,
		Description: description,
	}
}

type Knowledge struct {
	bun.BaseModel `bun:"table:knowledges,alias:k"`

	ID        uuid.UUID       `bun:"id,pk,type:uuid"`
	SourceID  uuid.UUID       `bun:"source_id,type:uuid,notnull"`
	Source    KnowledgeSource `bun:"rel:belongs-to,join:source_id=id"`
	CreatedAt bun.NullTime    `bun:"created_at,notnull,default:current_timestamp"`
	UpdatedAt bun.NullTime    `bun:"updated_at,notnull,default:current_timestamp"`

	Chunks []Chunk `bun:"rel:has-many,join:id=knowledge_id"`
}

func NewKnowledge(sourceID uuid.UUID) *Knowledge {
	return &Knowledge{
		ID:       uuid.New(),
		SourceID: sourceID,
	}
}

type Chunk struct {
	bun.BaseModel `bun:"table:chunks,alias:c"`

	ID          uuid.UUID `bun:"id,pk,type:uuid"`
	Index       int       `bun:"index,notnull"`
	KnowledgeID uuid.UUID `bun:"knowledge_id,type:uuid,notnull"`
	Knowledge   Knowledge `bun:"rel:belongs-to,join:knowledge_id=id"`
}

func NewChunk(knowledgeID uuid.UUID, index int) *Chunk {
	return &Chunk{
		ID:          uuid.New(),
		KnowledgeID: knowledgeID,
		Index:       index,
	}
}

type Embedding struct {
	bun.BaseModel `bun:"table:embeddings,alias:e"`

	ID        uuid.UUID `bun:"id,pk,type:uuid"`
	ChunkID   uuid.UUID `bun:"chunk_id,type:uuid,notnull"`
	Chunk     Chunk     `bun:"rel:has-one,join:chunk_id=id"`
	Embedding []float32 `bun:"embedding,type:vector(768),notnull"`
	Content   string    `bun:"content,type:text,notnull"`
}

func NewEmbedding(chunkID uuid.UUID, embedding []float32, content string) *Embedding {
	return &Embedding{
		ID:        uuid.New(),
		ChunkID:   chunkID,
		Embedding: embedding,
		Content:   content,
	}
}

type Image struct {
	bun.BaseModel `bun:"table:images,alias:i"`

	ID          uuid.UUID `bun:"id,pk,type:uuid"`
	KnowledgeID uuid.UUID `bun:"knowledge_id,type:uuid,notnull"`
	Knowledge   Knowledge `bun:"rel:belongs-to,join:knowledge_id=id"`
	Blob        string    `bun:"blob,type:text,notnull"`
}

func NewImage(knowledgeID uuid.UUID, blob string) *Image {
	return &Image{
		ID:          uuid.New(),
		KnowledgeID: knowledgeID,
		Blob:        blob,
	}
}
