/**
 * NobodyWho Design System components, ported to TSX from
 * ../NobodyWho Design System/components/ (kept 1:1 with the JSX originals).
 */
import type {
  AnchorHTMLAttributes,
  ButtonHTMLAttributes,
  HTMLAttributes,
  ReactNode,
} from "react";

/** The nobodywho wordmark — lowercase Inter Tight 600, tracked -0.035em. No drawn logo exists. */
export function Wordmark({
  size = 24,
  href,
  color,
  className = "",
  ...rest
}: {
  size?: number;
  href?: string;
  color?: string;
  className?: string;
} & AnchorHTMLAttributes<HTMLAnchorElement>) {
  const style = { fontSize: size, color: color || "inherit" };
  const cls = "nw-logo " + className;
  return href ? (
    <a className={cls} href={href} style={style} aria-label="NobodyWho, front page" {...rest}>
      nobodywho
    </a>
  ) : (
    <span className={cls} style={style} {...rest}>
      nobodywho
    </span>
  );
}

type ButtonProps = ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: "solid" | "outline";
  href?: string;
};

/** Square uppercase button. `solid` = accent fill (primary); `outline` = accent outline (secondary). */
export function Button({
  variant = "solid",
  href,
  children,
  onClick,
  type = "button",
  className = "",
  ...rest
}: ButtonProps) {
  const cls = ["nw-btn", variant === "solid" ? "nw-btn--solid" : "nw-btn--flat", className]
    .filter(Boolean)
    .join(" ");
  if (href) {
    const ext = href.startsWith("http");
    return (
      <a
        className={cls}
        href={href}
        onClick={onClick as unknown as React.MouseEventHandler<HTMLAnchorElement>}
        target={ext ? "_blank" : undefined}
        rel={ext ? "noopener" : undefined}
      >
        {children}
      </a>
    );
  }
  return (
    <button className={cls} type={type} onClick={onClick} {...rest}>
      {children}
    </button>
  );
}

/** Small uppercase section label (13px, 500, +0.06em), in the section's muted ink. */
export function Eyebrow({
  children,
  className = "",
  ...rest
}: {
  children: ReactNode;
  className?: string;
} & HTMLAttributes<HTMLParagraphElement>) {
  return (
    <p className={"nw-eyebrow " + className} {...rest}>
      {children}
    </p>
  );
}
