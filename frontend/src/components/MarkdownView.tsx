import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

/** 助手回复是 Markdown 源码。渲染成元素，而不是把 #、** 原样印出来。
 * 不用 innerHTML：react-markdown 默认不执行原始 HTML。 */
export function MarkdownView({ text }: { text: string }) {
  // Source ids are an internal RAG validation protocol; keep them in the
  // stored/API answer, but remove them at the presentation boundary so users
  // see a natural answer instead of implementation markers.
  const visibleText = text
    // Remove internal source ids emitted by the RAG citation protocol.
    .replace(/\s*\[S\d+\]/gi, "")
    // Models may also summarize the same evidence as a standalone source line.
    .replace(/^\s*来源\s*[:：].*$(?:\r?\n|$)/gim, "")
    .replace(/\n{3,}/g, "\n\n");

  return (
    <div className="md-view">
      <ReactMarkdown remarkPlugins={[remarkGfm]}>{visibleText}</ReactMarkdown>
    </div>
  );
}
