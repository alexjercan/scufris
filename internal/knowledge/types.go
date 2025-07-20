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
	bun.BaseModel `bun:"table:knowledge,alias:k"`

	ID        uuid.UUID       `bun:"id,pk,type:uuid"`
	SourceID  uuid.UUID       `bun:"source_id,type:uuid,notnull"`
	Source    KnowledgeSource `bun:"rel:belongs-to,join:source_id=id"`
	Content   string          `bun:"content,type:text,notnull"`
	CreatedAt bun.NullTime    `bun:"created_at,notnull,default:current_timestamp"`
	UpdatedAt bun.NullTime    `bun:"updated_at,notnull,default:current_timestamp"`

	Chunks []Chunk `bun:"rel:has-many,join:id=knowledge_id"`
}

func NewKnowledge(sourceID uuid.UUID, content string) *Knowledge {
	return &Knowledge{
		ID:       uuid.New(),
		SourceID: sourceID,
		Content:  content,
	}
}

type Chunk struct {
	bun.BaseModel `bun:"table:chunks,alias:c"`

	ID          uuid.UUID `bun:"id,pk,type:uuid"`
	Index       int       `bun:"index,notnull"`
	KnowledgeID uuid.UUID `bun:"knowledge_id,type:uuid,notnull"`
	Knowledge   Knowledge `bun:"rel:belongs-to,join:knowledge_id=id"`
	Content     string    `bun:"content,type:text,notnull"`
}

func NewChunk(knowledgeID uuid.UUID, index int, content string) *Chunk {
	return &Chunk{
		ID:          uuid.New(),
		KnowledgeID: knowledgeID,
		Index:       index,
		Content:     content,
	}
}

type Embedding struct {
	bun.BaseModel `bun:"table:embeddings,alias:e"`

	ID        uuid.UUID `bun:"id,pk,type:uuid"`
	ChunkID   uuid.UUID `bun:"chunk_id,type:uuid,notnull"`
	Chunk     Chunk     `bun:"rel:has-one,join:chunk_id=id"`
	Embedding []float32 `bun:"embedding,,type:vector(4096),notnull`
}

func NewEmbedding(chunkID uuid.UUID, embedding []float32) *Embedding {
	return &Embedding{
		ID:        uuid.New(),
		ChunkID:   chunkID,
		Embedding: embedding,
	}
}
