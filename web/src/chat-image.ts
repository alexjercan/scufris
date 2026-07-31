// Composer image attachment: read an image File into a data URL (for the inline
// preview + bubble) and the base64/mime payload the backend wants. The DOM
// (preview thumb, remove button) is built by the chat component; this module is
// just the file->payload read, kept small and reusable.

import type { ImageAttachment } from "./agent-types";

export interface PendingImage {
    dataUrl: string;
    attachment: ImageAttachment;
}

// Read an image File into a data URL + the base64/mime to send. Resolves null for
// a non-image file or an unreadable result (so the caller simply ignores it).
export function readImageFile(file: File): Promise<PendingImage | null> {
    if (!file.type.startsWith("image/")) return Promise.resolve(null);
    return new Promise((resolve) => {
        const reader = new FileReader();
        reader.onload = () => {
            const dataUrl =
                typeof reader.result === "string" ? reader.result : "";
            const comma = dataUrl.indexOf(",");
            if (comma < 0) {
                resolve(null);
                return;
            }
            resolve({
                dataUrl,
                attachment: {
                    data_base64: dataUrl.slice(comma + 1),
                    mime: file.type,
                },
            });
        };
        reader.onerror = () => resolve(null);
        reader.readAsDataURL(file);
    });
}
